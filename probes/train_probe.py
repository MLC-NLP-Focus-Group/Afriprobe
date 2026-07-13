from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


IGNORE_INDEX = -100


class LinearProbe(nn.Module):
    def __init__(self, hidden_dim: int, num_labels: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.classifier(hidden_states)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a linear POS probe on extracted hidden states.")
    parser.add_argument("--hidden_dir", default="outputs/hidden_states")
    parser.add_argument("--model_alias", required=True)
    parser.add_argument("--source_language", required=True)
    parser.add_argument("--target_language", default=None)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--eval_split", default="test")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output_dir", default="outputs/probes")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_arg)


def split_dir(hidden_dir: Path, model_alias: str, language: str, split: str) -> Path:
    path = hidden_dir / model_alias / language / split
    if not path.exists():
        raise FileNotFoundError(f"Missing hidden-state directory: {path}")
    return path


def chunk_files(path: Path) -> list[Path]:
    files = sorted(path.glob("chunk_*.pt"))
    if not files:
        raise FileNotFoundError(f"No chunk_*.pt files found in {path}")
    return files


def load_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_layer(chunk: dict[str, Any], layer: int, chunk_path: Path) -> None:
    hidden_states = chunk["hidden_states"]
    num_layers = hidden_states.shape[1]
    layer_slot = resolve_layer_slot(chunk, layer)
    if layer_slot < 0 or layer_slot >= num_layers:
        raise ValueError(f"Requested layer {layer}, but {chunk_path} has layers 0..{num_layers - 1}.")


def resolve_layer_slot(chunk: dict[str, Any], layer: int) -> int:
    saved_layers = chunk.get("metadata", {}).get("layers", "all")
    if saved_layers == "all":
        return layer
    if layer in saved_layers:
        return saved_layers.index(layer)
    raise ValueError(f"Requested original layer {layer}, but this chunk contains layers: {saved_layers}")


def flatten_valid_tokens(chunk: dict[str, Any], layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    validate_layer(chunk, layer, Path("<memory>"))
    layer_slot = resolve_layer_slot(chunk, layer)
    hidden_states = chunk["hidden_states"][:, layer_slot, :, :]
    labels = chunk["labels"]
    valid_positions = labels != IGNORE_INDEX
    return hidden_states[valid_positions].float(), labels[valid_positions].long()


def load_training_tokens(files: list[Path], layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_parts = []
    label_parts = []
    for file_path in tqdm(files, desc="loading train chunks", leave=False):
        chunk = torch.load(file_path, map_location="cpu")
        validate_layer(chunk, layer, file_path)
        hidden, labels = flatten_valid_tokens(chunk, layer)
        hidden_parts.append(hidden)
        label_parts.append(labels)

    if not hidden_parts:
        raise RuntimeError("No training tokens were loaded.")

    hidden = torch.cat(hidden_parts, dim=0)
    labels = torch.cat(label_parts, dim=0)
    print(f"[probe] train matrix: hidden={tuple(hidden.shape)}, labels={tuple(labels.shape)}")
    return hidden, labels


def load_cached_training_tokens(cache_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    cache = torch.load(cache_path, map_location="cpu")
    hidden = cache["hidden"]
    labels = cache["labels"]
    print(
        f"[probe] cached train matrix: hidden={tuple(hidden.shape)}, "
        f"labels={tuple(labels.shape)} from {cache_path}"
    )
    return hidden, labels


def macro_f1_from_counts(predictions: torch.Tensor, labels: torch.Tensor, num_labels: int) -> float:
    f1_scores = []
    for label_id in range(num_labels):
        predicted = predictions == label_id
        gold = labels == label_id
        true_positive = (predicted & gold).sum().item()
        false_positive = (predicted & ~gold).sum().item()
        false_negative = (~predicted & gold).sum().item()

        if true_positive == 0 and false_positive == 0 and false_negative == 0:
            continue

        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_scores.append(f1)

    return float(sum(f1_scores) / len(f1_scores)) if f1_scores else 0.0


def evaluate(
    probe: LinearProbe,
    files: list[Path],
    layer: int,
    batch_size: int,
    num_labels: int,
    device: torch.device,
    description: str,
) -> dict[str, float | int]:
    probe.eval()
    all_predictions = []
    all_labels = []
    total = 0
    correct = 0

    with torch.inference_mode():
        for file_path in tqdm(files, desc=description, leave=False):
            chunk = torch.load(file_path, map_location="cpu")
            validate_layer(chunk, layer, file_path)
            hidden, labels = flatten_valid_tokens(chunk, layer)
            if hidden.numel() == 0:
                continue

            for start in range(0, hidden.shape[0], batch_size):
                end = start + batch_size
                batch_hidden = hidden[start:end].to(device)
                batch_labels = labels[start:end].to(device)
                logits = probe(batch_hidden)
                predictions = logits.argmax(dim=-1)
                correct += (predictions == batch_labels).sum().item()
                total += batch_labels.numel()
                all_predictions.append(predictions.cpu())
                all_labels.append(batch_labels.cpu())

    if total == 0:
        raise RuntimeError(f"No valid tokens found during evaluation: {description}")

    predictions_tensor = torch.cat(all_predictions)
    labels_tensor = torch.cat(all_labels)
    return {
        "accuracy": correct / total,
        "macro_f1": macro_f1_from_counts(predictions_tensor, labels_tensor, num_labels),
        "num_tokens": total,
    }


def train_probe(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    hidden_dim: int,
    num_labels: int,
    batch_size: int,
    lr: float,
    epochs: int,
    device: torch.device,
) -> tuple[LinearProbe, list[dict[str, float]]]:
    probe = LinearProbe(hidden_dim=hidden_dim, num_labels=num_labels).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dataloader = DataLoader(TensorDataset(hidden, labels), batch_size=batch_size, shuffle=True)
    history = []

    for epoch in range(1, epochs + 1):
        probe.train()
        total_loss = 0.0
        total = 0
        correct = 0

        for batch_hidden, batch_labels in tqdm(dataloader, desc=f"epoch {epoch}", leave=False):
            batch_hidden = batch_hidden.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = probe(batch_hidden)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_labels.numel()
            predictions = logits.argmax(dim=-1)
            correct += (predictions == batch_labels).sum().item()
            total += batch_labels.numel()

        epoch_metrics = {
            "epoch": epoch,
            "loss": total_loss / total,
            "accuracy": correct / total,
        }
        history.append(epoch_metrics)
        print(
            f"[probe] epoch={epoch} loss={epoch_metrics['loss']:.4f} "
            f"accuracy={epoch_metrics['accuracy']:.4f}"
        )

    return probe, history


def save_outputs(
    output_dir: Path,
    model_alias: str,
    source_language: str,
    target_language: str,
    layer: int,
    probe: LinearProbe,
    metrics: dict[str, Any],
) -> Path:
    run_dir = output_dir / model_alias / f"{source_language}_to_{target_language}" / f"layer_{layer}"
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
    return run_dir


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    hidden_dir = Path(args.hidden_dir)
    target_language = args.target_language or args.source_language

    train_path = split_dir(hidden_dir, args.model_alias, args.source_language, args.train_split)
    source_eval_path = split_dir(hidden_dir, args.model_alias, args.source_language, args.eval_split)
    target_eval_path = split_dir(hidden_dir, args.model_alias, target_language, args.eval_split)

    train_manifest = load_manifest(train_path)
    target_manifest = load_manifest(target_eval_path)
    label_list = train_manifest["label_list"]
    num_labels = len(label_list)
    if target_manifest.get("label_list") != label_list:
        print("[probe] warning: source and target label lists differ; using source label order.")

    train_files = chunk_files(train_path)
    source_eval_files = chunk_files(source_eval_path)
    target_eval_files = chunk_files(target_eval_path)

    hidden, labels = load_training_tokens(train_files, args.layer)
    hidden_dim = hidden.shape[-1]
    print(
        f"[probe] model={args.model_alias} source={args.source_language} target={target_language} "
        f"layer={args.layer} hidden_dim={hidden_dim} num_labels={num_labels} device={device}"
    )

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

    source_eval = evaluate(
        probe=probe,
        files=source_eval_files,
        layer=args.layer,
        batch_size=args.batch_size,
        num_labels=num_labels,
        device=device,
        description=f"eval {args.source_language}/{args.eval_split}",
    )
    target_eval = evaluate(
        probe=probe,
        files=target_eval_files,
        layer=args.layer,
        batch_size=args.batch_size,
        num_labels=num_labels,
        device=device,
        description=f"eval {target_language}/{args.eval_split}",
    )

    metrics: dict[str, Any] = {
        "model_alias": args.model_alias,
        "source_language": args.source_language,
        "target_language": target_language,
        "layer": args.layer,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "hidden_dim": hidden_dim,
        "num_labels": num_labels,
        "label_list": label_list,
        "train_history": train_history,
        "source_eval": source_eval,
        "target_eval": target_eval,
    }
    run_dir = save_outputs(
        output_dir=Path(args.output_dir),
        model_alias=args.model_alias,
        source_language=args.source_language,
        target_language=target_language,
        layer=args.layer,
        probe=probe,
        metrics=metrics,
    )

    print(f"[probe] source_eval accuracy={source_eval['accuracy']:.4f} macro_f1={source_eval['macro_f1']:.4f}")
    print(f"[probe] target_eval accuracy={target_eval['accuracy']:.4f} macro_f1={target_eval['macro_f1']:.4f}")
    print(f"[probe] saved checkpoint and metrics to {run_dir}")


if __name__ == "__main__":
    main()
