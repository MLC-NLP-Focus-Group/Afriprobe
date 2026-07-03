from __future__ import annotations

import argparse
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
) -> None:
    dataset = MasakhaPOSDataset(
        tokenizer_name_or_path=model_name_or_path,
        tokenizer=tokenizer,
        language=language,
        split=split,
        max_length=max_length,
        dataset_name=dataset_name,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_examples)
    split_dir = output_dir / model_alias / language / split
    split_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "model_name_or_path": model_name_or_path,
        "model_alias": model_alias,
        "dataset_name": dataset_name,
        "language": language,
        "split": split,
        "max_length": max_length,
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
            hidden_states = torch.stack(outputs.hidden_states, dim=1).cpu()
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
                    f"labels={tuple(labels.shape)}, attention_mask={tuple(mask.shape)}"
                )

    save_manifest(split_dir, manifest)
    print(f"[extract] saved {len(manifest['chunks'])} chunks to {split_dir}")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)

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
                )
            except Exception as exc:
                raise RuntimeError(f"Failed while extracting {language}/{split}.") from exc


if __name__ == "__main__":
    main()
