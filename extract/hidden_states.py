from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "datasets"
sys.path.insert(0, str(DATASET_DIR))

from masakhapos import MasakhaPOSDataset  # noqa: E402


def validate_dataset_import() -> None:
    signature = inspect.signature(MasakhaPOSDataset.__init__)
    if "tokenizer_name_or_path" not in signature.parameters:
        module_path = inspect.getfile(MasakhaPOSDataset)
        raise RuntimeError(
            "Imported an incompatible MasakhaPOSDataset. "
            f"Loaded from: {module_path}. "
            "Expected datasets/masakhapos.py with a tokenizer_name_or_path argument. "
            "Delete stale duplicate dataset files and rerun."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen encoder hidden states for MasakhaPOS.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--model_alias", required=True)
    parser.add_argument("--languages", nargs="+", required=True)
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--output_dir", default="outputs/hidden_states")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset_name", default="masakhapos")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--github_repo", default="https://raw.githubusercontent.com/masakhane-io/masakhane-pos")
    parser.add_argument("--save_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--layers", nargs="+", default=None, help="Layer indices to save. Defaults to all layers.")
    parser.add_argument("--overwrite", action="store_true", help="Re-extract splits even when a manifest already exists.")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_arg)


def collate_examples(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "input_ids": torch.stack([example["input_ids"] for example in batch]),
        "attention_mask": torch.stack([example["attention_mask"] for example in batch]),
        "labels": torch.stack([example["labels"] for example in batch]),
        "tokens": [example["tokens"] for example in batch],
        "word_ids": [example["word_ids"] for example in batch],
    }


def save_manifest(split_dir: Path, manifest: dict[str, Any]) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    with (split_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def completed_split_exists(split_dir: Path) -> bool:
    manifest_path = split_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except json.JSONDecodeError:
        return False
    chunks = manifest.get("chunks", [])
    return bool(chunks) and all((split_dir / chunk["file"]).exists() for chunk in chunks)


def parse_layer_indices(layer_args: list[str] | None) -> list[int] | None:
    if layer_args is None:
        return None
    layers: list[int] = []
    for layer_arg in layer_args:
        for part in layer_arg.split(","):
            part = part.strip()
            if part:
                layers.append(int(part))
    return layers


def resolve_save_dtype(save_dtype: str) -> torch.dtype:
    if save_dtype == "float16":
        return torch.float16
    if save_dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported save_dtype: {save_dtype}")


def extract_split(
    model: torch.nn.Module,
    tokenizer: Any,
    model_name_or_path: str,
    model_alias: str,
    language: str,
    split: str,
    batch_size: int,
    max_length: int,
    output_dir: Path,
    device: torch.device,
    dataset_name: str,
    data_dir: str | None,
    github_repo: str,
    save_dtype: torch.dtype,
    layer_indices: list[int] | None,
    overwrite: bool,
) -> None:
    split_dir = output_dir / model_alias / language / split
    if not overwrite and completed_split_exists(split_dir):
        print(f"[extract] skipping completed split: {split_dir}")
        return

    dataset = MasakhaPOSDataset(
        tokenizer_name_or_path=model_name_or_path,
        tokenizer=tokenizer,
        language=language,
        split=split,
        max_length=max_length,
        dataset_name=dataset_name,
        data_dir=data_dir,
        github_repo=github_repo,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_examples)
    split_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "model_name_or_path": model_name_or_path,
        "model_alias": model_alias,
        "dataset_name": dataset_name,
        "data_dir": data_dir,
        "github_repo": github_repo,
        "language": language,
        "split": split,
        "max_length": max_length,
        "save_dtype": str(save_dtype).replace("torch.", ""),
        "layers": layer_indices if layer_indices is not None else "all",
        "num_examples": len(dataset),
        "label_list": dataset.label_list,
        "chunks": [],
    }

    print(
        f"[extract] {language}/{split}: {len(dataset)} examples, "
        f"batch_size={batch_size}, max_length={max_length}, labels={len(dataset.label_list)}"
    )

    with torch.inference_mode():
        for chunk_index, batch in enumerate(tqdm(dataloader, desc=f"{language}/{split}", leave=False)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_state_layers = outputs.hidden_states
            if layer_indices is not None:
                hidden_state_layers = tuple(hidden_state_layers[layer_index] for layer_index in layer_indices)
            hidden_states = torch.stack(hidden_state_layers, dim=1).to(dtype=save_dtype).cpu()
            labels = batch["labels"].cpu()
            mask = batch["attention_mask"].cpu()

            chunk_name = f"chunk_{chunk_index:04d}.pt"
            chunk_path = split_dir / chunk_name
            torch.save(
                {
                    "hidden_states": hidden_states,
                    "labels": labels,
                    "attention_mask": mask,
                    "metadata": {
                        "model_name_or_path": model_name_or_path,
                        "model_alias": model_alias,
                        "language": language,
                        "split": split,
                        "max_length": max_length,
                        "save_dtype": str(save_dtype).replace("torch.", ""),
                        "layers": layer_indices if layer_indices is not None else "all",
                        "label_list": dataset.label_list,
                        "tokens": batch["tokens"],
                        "word_ids": batch["word_ids"],
                    },
                },
                chunk_path,
            )

            manifest["chunks"].append(
                {
                    "file": chunk_name,
                    "hidden_states_shape": list(hidden_states.shape),
                    "labels_shape": list(labels.shape),
                    "attention_mask_shape": list(mask.shape),
                }
            )

            if chunk_index == 0:
                print(
                    f"[extract] first chunk shape: hidden_states={tuple(hidden_states.shape)}, "
                    f"labels={tuple(labels.shape)}, attention_mask={tuple(mask.shape)}, dtype={hidden_states.dtype}"
                )

    save_manifest(split_dir, manifest)
    print(f"[extract] saved {len(manifest['chunks'])} chunks to {split_dir}")


def main() -> None:
    validate_dataset_import()
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    save_dtype = resolve_save_dtype(args.save_dtype)
    layer_indices = parse_layer_indices(args.layers)

    print(f"[extract] loading tokenizer/model: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModel.from_pretrained(args.model_name_or_path)
    model.requires_grad_(False)
    model.eval()
    model.to(device)
    print(f"[extract] model on {device}; encoder parameters frozen")

    for language in args.languages:
        for split in args.splits:
            try:
                extract_split(
                    model=model,
                    tokenizer=tokenizer,
                    model_name_or_path=args.model_name_or_path,
                    model_alias=args.model_alias,
                    language=language,
                    split=split,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                    output_dir=output_dir,
                    device=device,
                    dataset_name=args.dataset_name,
                    data_dir=args.data_dir,
                    github_repo=args.github_repo,
                    save_dtype=save_dtype,
                    layer_indices=layer_indices,
                    overwrite=args.overwrite,
                )
            except Exception as exc:
                raise RuntimeError(f"Failed while extracting {language}/{split}.") from exc


if __name__ == "__main__":
    main()
