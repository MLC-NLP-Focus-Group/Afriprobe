from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize layer_sweep_metrics.json files into CSV tables.")
    parser.add_argument("--input_dir", required=True, help="Directory containing layer_sweep_metrics.json files.")
    parser.add_argument("--output_dir", default="analysis/probe_results")
    parser.add_argument("--model_alias", default=None, help="Optional model alias filter, e.g. xlmr.")
    parser.add_argument("--languages", nargs="+", default=None, help="Optional language order for matrix CSVs.")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def find_summary_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.rglob("layer_sweep_metrics.json"))
    if not files:
        raise FileNotFoundError(f"No layer_sweep_metrics.json files found under {input_dir}")
    return files


def flatten_summary(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    model_alias = summary["model_alias"]
    source_language = summary["source_language"]
    train_split = summary.get("train_split", "train")
    eval_split = summary.get("eval_split", "test")
    rows = []

    for result in summary.get("results", []):
        layer = int(result["layer"])
        for target_language, metrics in result.get("target_eval", {}).items():
            accuracy = float(metrics["accuracy"])
            macro_f1 = float(metrics["macro_f1"])
            rows.append(
                {
                    "model": model_alias,
                    "source_language": source_language,
                    "target_language": target_language,
                    "layer": layer,
                    "is_self_transfer": source_language == target_language,
                    "accuracy": accuracy,
                    "accuracy_pct": accuracy * 100,
                    "macro_f1": macro_f1,
                    "macro_f1_pct": macro_f1 * 100,
                    "num_tokens": int(metrics.get("num_tokens", 0)),
                    "train_split": train_split,
                    "eval_split": eval_split,
                    "input_file": path.name,
                }
            )

    return rows


def best_layer_by_pair(long_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in long_rows:
        grouped[(row["model"], row["source_language"], row["target_language"])].append(row)

    best_rows = []
    for (model, source, target), rows in sorted(grouped.items()):
        best_accuracy = max(rows, key=lambda row: row["accuracy"])
        best_macro_f1 = max(rows, key=lambda row: row["macro_f1"])
        best_rows.append(
            {
                "model": model,
                "source_language": source,
                "target_language": target,
                "is_self_transfer": source == target,
                "best_accuracy_layer": best_accuracy["layer"],
                "best_accuracy": best_accuracy["accuracy"],
                "best_accuracy_pct": best_accuracy["accuracy_pct"],
                "best_macro_f1_layer": best_macro_f1["layer"],
                "best_macro_f1": best_macro_f1["macro_f1"],
                "best_macro_f1_pct": best_macro_f1["macro_f1_pct"],
                "num_layers": len({row["layer"] for row in rows}),
            }
        )
    return best_rows


def transfer_matrix(
    best_rows: list[dict[str, Any]],
    languages: list[str],
    value_key: str,
) -> list[dict[str, Any]]:
    values = {
        (row["source_language"], row["target_language"]): row[value_key]
        for row in best_rows
    }
    matrix_rows = []
    for source in languages:
        row = {"source_target": source}
        for target in languages:
            value = values.get((source, target), "")
            row[target] = f"{value:.6f}" if value != "" else ""
        matrix_rows.append(row)
    return matrix_rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def layer_average_summary(long_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in long_rows:
        grouped[row["layer"]].append(row)

    rows = []
    for layer, layer_rows in sorted(grouped.items()):
        cross_rows = [row for row in layer_rows if not row["is_self_transfer"]]
        self_rows = [row for row in layer_rows if row["is_self_transfer"]]
        rows.append(
            {
                "layer": layer,
                "mean_accuracy_all": mean([row["accuracy"] for row in layer_rows]),
                "mean_accuracy_cross_lingual": mean([row["accuracy"] for row in cross_rows]),
                "mean_accuracy_self": mean([row["accuracy"] for row in self_rows]),
                "mean_macro_f1_all": mean([row["macro_f1"] for row in layer_rows]),
                "mean_macro_f1_cross_lingual": mean([row["macro_f1"] for row in cross_rows]),
                "mean_macro_f1_self": mean([row["macro_f1"] for row in self_rows]),
                "num_pairs_all": len(layer_rows),
                "num_pairs_cross_lingual": len(cross_rows),
            }
        )
    return rows


def source_summary(best_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in best_rows:
        grouped[row["source_language"]].append(row)

    rows = []
    for source, source_rows in sorted(grouped.items()):
        self_rows = [row for row in source_rows if row["is_self_transfer"]]
        cross_rows = [row for row in source_rows if not row["is_self_transfer"]]
        best_cross = max(cross_rows, key=lambda row: row["best_accuracy"]) if cross_rows else None
        self_row = self_rows[0] if self_rows else {}
        rows.append(
            {
                "source_language": source,
                "self_best_accuracy": self_row.get("best_accuracy", ""),
                "self_best_macro_f1": self_row.get("best_macro_f1", ""),
                "mean_cross_best_accuracy": mean([row["best_accuracy"] for row in cross_rows]),
                "mean_cross_best_macro_f1": mean([row["best_macro_f1"] for row in cross_rows]),
                "best_cross_target_by_accuracy": best_cross["target_language"] if best_cross else "",
                "best_cross_accuracy": best_cross["best_accuracy"] if best_cross else "",
            }
        )
    return rows


def target_summary(best_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in best_rows:
        grouped[row["target_language"]].append(row)

    rows = []
    for target, target_rows in sorted(grouped.items()):
        cross_rows = [row for row in target_rows if not row["is_self_transfer"]]
        best_source = max(cross_rows, key=lambda row: row["best_accuracy"]) if cross_rows else None
        rows.append(
            {
                "target_language": target,
                "mean_incoming_cross_best_accuracy": mean([row["best_accuracy"] for row in cross_rows]),
                "mean_incoming_cross_best_macro_f1": mean([row["best_macro_f1"] for row in cross_rows]),
                "best_source_by_accuracy": best_source["source_language"] if best_source else "",
                "best_incoming_accuracy": best_source["best_accuracy"] if best_source else "",
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    long_rows = []
    for summary_path in find_summary_files(input_dir):
        rows = flatten_summary(summary_path)
        if args.model_alias is not None:
            rows = [row for row in rows if row["model"] == args.model_alias]
        long_rows.extend(rows)

    if not long_rows:
        raise RuntimeError("No probe result rows were loaded. Check --input_dir and --model_alias.")

    long_rows = sorted(
        long_rows,
        key=lambda row: (row["model"], row["source_language"], row["target_language"], row["layer"]),
    )
    best_rows = best_layer_by_pair(long_rows)
    languages = args.languages or sorted(
        {row["source_language"] for row in long_rows} | {row["target_language"] for row in long_rows}
    )

    write_csv(
        output_dir / "probe_transfer_long.csv",
        long_rows,
        [
            "model",
            "source_language",
            "target_language",
            "layer",
            "is_self_transfer",
            "accuracy",
            "accuracy_pct",
            "macro_f1",
            "macro_f1_pct",
            "num_tokens",
            "train_split",
            "eval_split",
            "input_file",
        ],
    )
    write_csv(
        output_dir / "best_layer_by_pair.csv",
        best_rows,
        [
            "model",
            "source_language",
            "target_language",
            "is_self_transfer",
            "best_accuracy_layer",
            "best_accuracy",
            "best_accuracy_pct",
            "best_macro_f1_layer",
            "best_macro_f1",
            "best_macro_f1_pct",
            "num_layers",
        ],
    )
    write_csv(
        output_dir / "best_accuracy_transfer_matrix.csv",
        transfer_matrix(best_rows, languages, "best_accuracy"),
        ["source_target", *languages],
    )
    write_csv(
        output_dir / "best_macro_f1_transfer_matrix.csv",
        transfer_matrix(best_rows, languages, "best_macro_f1"),
        ["source_target", *languages],
    )
    write_csv(
        output_dir / "layer_average_summary.csv",
        layer_average_summary(long_rows),
        [
            "layer",
            "mean_accuracy_all",
            "mean_accuracy_cross_lingual",
            "mean_accuracy_self",
            "mean_macro_f1_all",
            "mean_macro_f1_cross_lingual",
            "mean_macro_f1_self",
            "num_pairs_all",
            "num_pairs_cross_lingual",
        ],
    )
    write_csv(
        output_dir / "source_summary.csv",
        source_summary(best_rows),
        [
            "source_language",
            "self_best_accuracy",
            "self_best_macro_f1",
            "mean_cross_best_accuracy",
            "mean_cross_best_macro_f1",
            "best_cross_target_by_accuracy",
            "best_cross_accuracy",
        ],
    )
    write_csv(
        output_dir / "target_summary.csv",
        target_summary(best_rows),
        [
            "target_language",
            "mean_incoming_cross_best_accuracy",
            "mean_incoming_cross_best_macro_f1",
            "best_source_by_accuracy",
            "best_incoming_accuracy",
        ],
    )

    print(f"Wrote probe summary CSVs to {output_dir}")
    for path in sorted(output_dir.glob("*.csv")):
        print(path)


if __name__ == "__main__":
    main()
