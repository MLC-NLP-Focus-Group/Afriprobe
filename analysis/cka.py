from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBES_DIR = REPO_ROOT / "probes"
sys.path.insert(0, str(PROBES_DIR))

from train_probe import IGNORE_INDEX, chunk_files, load_manifest, resolve_layer_slot, split_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute pairwise linear CKA over saved hidden states.")
    parser.add_argument("--hidden_dir", default="outputs/hidden_states")
    parser.add_argument("--model_alias", required=True)
    parser.add_argument("--languages", nargs="+", required=True)
    parser.add_argument("--layers", nargs="+", default=["all"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_tokens", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_dir", default="outputs/analysis/cka")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_arg)


def parse_layer_args(layer_args: list[str], hidden_dir: Path, model_alias: str, language: str, split: str) -> list[int]:
    split_path = split_dir(hidden_dir, model_alias, language, split)
    first_chunk = torch.load(chunk_files(split_path)[0], map_location="cpu")
    saved_layers = first_chunk.get("metadata", {}).get("layers", "all")

    if layer_args == ["all"]:
        if saved_layers == "all":
            return list(range(first_chunk["hidden_states"].shape[1]))
        return [int(layer) for layer in saved_layers]

    layers = []
    for layer_arg in layer_args:
        for part in layer_arg.split(","):
            part = part.strip()
            if part:
                layer = int(part)
                resolve_layer_slot(first_chunk, layer)
                layers.append(layer)
    return layers


def linear_cka(left: torch.Tensor, right: torch.Tensor, device: torch.device) -> float:
    left = left.to(device=device, dtype=torch.float32)
    right = right.to(device=device, dtype=torch.float32)

    left = left - left.mean(dim=0, keepdim=True)
    right = right - right.mean(dim=0, keepdim=True)

    hsic = torch.linalg.matrix_norm(left.T @ right, ord="fro") ** 2
    left_norm = torch.linalg.matrix_norm(left.T @ left, ord="fro")
    right_norm = torch.linalg.matrix_norm(right.T @ right, ord="fro")
    score = hsic / (left_norm * right_norm + 1e-8)
    return float(score.detach().cpu())


def sample_rows(matrix: torch.Tensor, max_tokens: int, seed: int) -> torch.Tensor:
    if matrix.shape[0] <= max_tokens:
        return matrix
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(matrix.shape[0], generator=generator)[:max_tokens]
    return matrix[indices]


def load_language_layer_tokens(
    hidden_dir: Path,
    model_alias: str,
    language: str,
    split: str,
    layer: int,
    max_tokens: int,
    seed: int,
) -> torch.Tensor:
    path = split_dir(hidden_dir, model_alias, language, split)
    files = chunk_files(path)
    parts = []
    total_valid = 0

    for file_path in tqdm(files, desc=f"load {language}/{split}/layer_{layer}", leave=False):
        chunk = torch.load(file_path, map_location="cpu")
        layer_slot = resolve_layer_slot(chunk, layer)
        hidden_states = chunk["hidden_states"][:, layer_slot, :, :]
        labels = chunk["labels"]
        valid_positions = labels != IGNORE_INDEX
        valid_hidden = hidden_states[valid_positions].float()
        total_valid += valid_hidden.shape[0]
        parts.append(valid_hidden)

    if not parts:
        raise RuntimeError(f"No hidden-state chunks found for {language}/{split}.")

    matrix = torch.cat(parts, dim=0)
    if matrix.shape[0] == 0:
        raise RuntimeError(f"No valid POS-token positions found for {language}/{split}.")

    sampled = sample_rows(matrix, max_tokens=max_tokens, seed=seed)
    print(
        f"[cka] {language} layer={layer}: valid_tokens={total_valid}, "
        f"sampled={sampled.shape[0]}, hidden_dim={sampled.shape[1]}"
    )
    return sampled


def compute_layer_matrix(
    representations: dict[str, torch.Tensor],
    languages: list[str],
    device: torch.device,
) -> list[list[float]]:
    matrix = []
    for left_language in languages:
        row = []
        for right_language in languages:
            left = representations[left_language]
            right = representations[right_language]
            n_samples = min(left.shape[0], right.shape[0])
            row.append(linear_cka(left[:n_samples], right[:n_samples], device=device))
        matrix.append(row)
    return matrix


def save_layer_outputs(
    output_dir: Path,
    model_alias: str,
    split: str,
    layer: int,
    languages: list[str],
    matrix: list[list[float]],
    metadata: dict[str, Any],
) -> None:
    layer_dir = output_dir / model_alias / split / f"layer_{layer}"
    layer_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_alias": model_alias,
        "split": split,
        "layer": layer,
        "languages": languages,
        "matrix": matrix,
        "metadata": metadata,
    }
    with (layer_dir / "cka_matrix.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    with (layer_dir / "cka_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["language", *languages])
        for language, row in zip(languages, matrix):
            writer.writerow([language, *row])


def save_summary(output_dir: Path, model_alias: str, split: str, summary: dict[str, Any]) -> None:
    summary_dir = output_dir / model_alias / split
    summary_dir.mkdir(parents=True, exist_ok=True)
    with (summary_dir / "cka_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    args = parse_args()
    hidden_dir = Path(args.hidden_dir)
    output_dir = Path(args.output_dir)
    device = resolve_device(args.device)
    layers = parse_layer_args(args.layers, hidden_dir, args.model_alias, args.languages[0], args.split)

    for language in args.languages:
        manifest_path = split_dir(hidden_dir, args.model_alias, language, args.split)
        load_manifest(manifest_path)

    summary: dict[str, Any] = {
        "model_alias": args.model_alias,
        "split": args.split,
        "languages": args.languages,
        "layers": layers,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "results": [],
    }

    print(
        f"[cka] model={args.model_alias} split={args.split} "
        f"languages={args.languages} layers={layers} max_tokens={args.max_tokens}"
    )

    for layer in layers:
        representations = {
            language: load_language_layer_tokens(
                hidden_dir=hidden_dir,
                model_alias=args.model_alias,
                language=language,
                split=args.split,
                layer=layer,
                max_tokens=args.max_tokens,
                seed=args.seed + index,
            )
            for index, language in enumerate(args.languages)
        }
        matrix = compute_layer_matrix(representations, args.languages, device)
        metadata = {
            language: {
                "num_sampled_tokens": representations[language].shape[0],
                "hidden_dim": representations[language].shape[1],
            }
            for language in args.languages
        }
        save_layer_outputs(output_dir, args.model_alias, args.split, layer, args.languages, matrix, metadata)
        summary["results"].append(
            {
                "layer": layer,
                "matrix_json": str(output_dir / args.model_alias / args.split / f"layer_{layer}" / "cka_matrix.json"),
                "matrix_csv": str(output_dir / args.model_alias / args.split / f"layer_{layer}" / "cka_matrix.csv"),
                "matrix": matrix,
            }
        )
        print(f"[cka] saved layer {layer} CKA matrix")

    save_summary(output_dir, args.model_alias, args.split, summary)
    print(f"[cka] saved summary to {output_dir / args.model_alias / args.split / 'cka_summary.json'}")


if __name__ == "__main__":
    main()
