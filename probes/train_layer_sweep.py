from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from train_probe import (
    LinearProbe,
    chunk_files,
    evaluate,
    load_manifest,
    load_training_tokens,
    resolve_device,
    resolve_layer_slot,
    split_dir,
    train_probe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train layer-wise linear probes for one source language.")
    parser.add_argument("--hidden_dir", default="outputs/hidden_states")
    parser.add_argument("--model_alias", required=True)
    parser.add_argument("--source_language", required=True)
    parser.add_argument("--target_languages", nargs="+", default=None)
    parser.add_argument("--layers", nargs="+", default=["all"])
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--eval_split", default="test")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output_dir", default="outputs/probes")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_layer_args(layer_args: list[str], train_files: list[Path]) -> list[int]:
    first_chunk = torch.load(train_files[0], map_location="cpu")
    saved_layers = first_chunk.get("metadata", {}).get("layers", "all")

    if layer_args == ["all"]:
        if saved_layers == "all":
            return list(range(first_chunk["hidden_states"].shape[1]))
        return [int(layer) for layer in saved_layers]

    requested_layers = []
    for layer_arg in layer_args:
        for part in layer_arg.split(","):
            part = part.strip()
            if part:
                requested_layers.append(int(part))

    for layer in requested_layers:
        resolve_layer_slot(first_chunk, layer)
    return requested_layers


def load_eval_sets(
    hidden_dir: Path,
    model_alias: str,
    target_languages: list[str],
    eval_split: str,
    source_label_list: list[str],
) -> dict[str, dict[str, Any]]:
    eval_sets = {}
    for target_language in target_languages:
        target_path = split_dir(hidden_dir, model_alias, target_language, eval_split)
        target_manifest = load_manifest(target_path)
        if target_manifest.get("label_list") != source_label_list:
            print(f"[sweep] warning: label list differs for target={target_language}; using source label order.")
        eval_sets[target_language] = {
            "path": target_path,
            "files": chunk_files(target_path),
            "manifest": target_manifest,
        }
    return eval_sets


def save_layer_outputs(
    run_dir: Path,
    probe: LinearProbe,
    metrics: dict[str, Any],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": probe.state_dict(),
            "hidden_dim": probe.classifier.in_features,
            "num_labels": probe.classifier.out_features,
            "metrics": metrics,
        },
        run_dir / "probe.pt",
    )
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def save_summary(summary_path: Path, summary: dict[str, Any]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    hidden_dir = Path(args.hidden_dir)
    output_root = Path(args.output_dir) / args.model_alias / f"source_{args.source_language}"
    summary_path = output_root / "layer_sweep_metrics.json"

    target_languages = args.target_languages or [args.source_language]
    if args.source_language not in target_languages:
        target_languages = [args.source_language, *target_languages]

    train_path = split_dir(hidden_dir, args.model_alias, args.source_language, args.train_split)
    train_manifest = load_manifest(train_path)
    train_files = chunk_files(train_path)
    label_list = train_manifest["label_list"]
    num_labels = len(label_list)
    layers = parse_layer_args(args.layers, train_files)
    eval_sets = load_eval_sets(
        hidden_dir=hidden_dir,
        model_alias=args.model_alias,
        target_languages=target_languages,
        eval_split=args.eval_split,
        source_label_list=label_list,
    )

    print(
        f"[sweep] model={args.model_alias} source={args.source_language} "
        f"targets={target_languages} layers={layers} device={device}"
    )

    summary: dict[str, Any] = {
        "model_alias": args.model_alias,
        "source_language": args.source_language,
        "target_languages": target_languages,
        "layers": layers,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "label_list": label_list,
        "results": [],
    }

    if summary_path.exists() and not args.overwrite:
        with summary_path.open("r", encoding="utf-8") as handle:
            existing_summary = json.load(handle)
        summary["results"] = existing_summary.get("results", [])

    completed_layers = {result["layer"] for result in summary["results"]}

    for layer in layers:
        run_dir = output_root / f"layer_{layer}"
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists() and layer in completed_layers and not args.overwrite:
            print(f"[sweep] skipping completed layer {layer}: {run_dir}")
            continue

        print(f"[sweep] training layer {layer}")
        hidden, labels = load_training_tokens(train_files, layer)
        hidden_dim = hidden.shape[-1]
        probe, train_history = train_probe(
            hidden=hidden,
            labels=labels,
            hidden_dim=hidden_dim,
            num_labels=num_labels,
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs,
            device=device,
        )

        target_metrics = {}
        for target_language, eval_set in eval_sets.items():
            target_metrics[target_language] = evaluate(
                probe=probe,
                files=eval_set["files"],
                layer=layer,
                batch_size=args.batch_size,
                num_labels=num_labels,
                device=device,
                description=f"eval {target_language}/{args.eval_split}/layer_{layer}",
            )
            scores = target_metrics[target_language]
            print(
                f"[sweep] layer={layer} target={target_language} "
                f"accuracy={scores['accuracy']:.4f} macro_f1={scores['macro_f1']:.4f}"
            )

        metrics = {
            "model_alias": args.model_alias,
            "source_language": args.source_language,
            "target_languages": target_languages,
            "layer": layer,
            "train_split": args.train_split,
            "eval_split": args.eval_split,
            "hidden_dim": hidden_dim,
            "num_labels": num_labels,
            "label_list": label_list,
            "train_history": train_history,
            "target_eval": target_metrics,
        }
        save_layer_outputs(run_dir, probe, metrics)

        summary["results"] = [result for result in summary["results"] if result["layer"] != layer]
        summary["results"].append(
            {
                "layer": layer,
                "checkpoint": str(run_dir / "probe.pt"),
                "metrics": str(run_dir / "metrics.json"),
                "target_eval": target_metrics,
            }
        )
        summary["results"] = sorted(summary["results"], key=lambda result: result["layer"])
        save_summary(summary_path, summary)

    print(f"[sweep] saved summary to {summary_path}")


if __name__ == "__main__":
    main()
