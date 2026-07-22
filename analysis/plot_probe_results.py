from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


REQUIRED_COLUMNS = ["model_alias", "source_language", "target_language", "layer", "accuracy", "macro_f1"]
COLUMN_ALIASES = {"model": "model_alias"}

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

LANGUAGE_COLORS = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"]

SEQUENTIAL_STEPS = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list("afriprobe_blues", SEQUENTIAL_STEPS)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "text.color": INK_PRIMARY,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "axes.grid": True,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "svg.fonttype": "none",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SVG figures from summarized probe CSV files.")
    parser.add_argument("--input_dir", default="analysis/probe_results", help="Directory containing probe summary CSVs.")
    parser.add_argument("--output_dir", default="analysis/probe_results/figures")
    parser.add_argument("--top_k", type=int, default=12, help="Number of cross-lingual pairs in the top-pairs figure.")
    parser.add_argument(
        "--probes_dir",
        default=None,
        help="Optional probe output directory (from train_layer_sweep.py). "
        "Summarizes layer_sweep_metrics.json files into a CSV in --input_dir before plotting.",
    )
    return parser.parse_args()


def summarize_probe_outputs(probes_dir: Path, input_dir: Path) -> None:
    sweep_paths = sorted(probes_dir.glob("*/source_*/layer_sweep_metrics.json"))
    if not sweep_paths:
        raise FileNotFoundError(f"No layer_sweep_metrics.json files found under {probes_dir}")

    rows = []
    for sweep_path in sweep_paths:
        with sweep_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        for result in summary["results"]:
            for target_language, scores in result["target_eval"].items():
                rows.append(
                    {
                        "model_alias": summary["model_alias"],
                        "source_language": summary["source_language"],
                        "target_language": target_language,
                        "layer": result["layer"],
                        "accuracy": scores["accuracy"],
                        "macro_f1": scores["macro_f1"],
                        "num_tokens": scores.get("num_tokens"),
                    }
                )

    frame = pd.DataFrame(rows).sort_values(["model_alias", "source_language", "target_language", "layer"])
    input_dir.mkdir(parents=True, exist_ok=True)
    for model_alias, model_frame in frame.groupby("model_alias"):
        csv_path = input_dir / f"{model_alias}_probe_summary.csv"
        model_frame.to_csv(csv_path, index=False)
        print(f"[plot] summarized {len(model_frame)} rows from {len(sweep_paths)} sweeps -> {csv_path}")


def load_summary_frames(input_dir: Path) -> pd.DataFrame:
    csv_paths = sorted(input_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(
            f"No CSV files found in {input_dir}. "
            "Run with --probes_dir pointing at the train_layer_sweep.py output directory to build them."
        )

    frames = []
    skipped = []
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path).rename(columns=COLUMN_ALIASES)
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            skipped.append(csv_path.name)
            continue
        frames.append(frame[[column for column in frame.columns if column in REQUIRED_COLUMNS or column == "num_tokens"]])
        print(f"[plot] loaded {len(frame)} rows from {csv_path.name}")

    if skipped:
        print(f"[plot] skipped {len(skipped)} CSVs without per-layer probe columns: {skipped}")
    if not frames:
        raise RuntimeError(
            f"No CSVs in {input_dir} contain the required columns: {REQUIRED_COLUMNS} "
            f"(a 'model' column is accepted in place of 'model_alias')"
        )
    return pd.concat(frames, ignore_index=True)


def language_order(frame: pd.DataFrame) -> list[str]:
    ordered = list(dict.fromkeys(frame["source_language"]))
    for target in dict.fromkeys(frame["target_language"]):
        if target not in ordered:
            ordered.append(target)
    return ordered


def language_color_map(languages: list[str]) -> dict[str, str]:
    if len(languages) > len(LANGUAGE_COLORS):
        raise ValueError(f"More languages ({len(languages)}) than fixed palette slots ({len(LANGUAGE_COLORS)}).")
    return {language: LANGUAGE_COLORS[index] for index, language in enumerate(languages)}


def figure_path(output_dir: Path, model_alias: str, name: str, multiple_models: bool) -> Path:
    if multiple_models:
        return output_dir / f"{model_alias}_{name}"
    return output_dir / name


def save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    print(f"[plot] wrote {path}")


def plot_transfer_heatmap(frame: pd.DataFrame, metric: str, languages: list[str], title: str, path: Path) -> None:
    best = frame.groupby(["source_language", "target_language"])[metric].max().unstack()
    best = best.reindex(index=languages, columns=languages)

    figure, axis = plt.subplots(figsize=(1.1 * len(languages) + 2.4, 1.0 * len(languages) + 1.8))
    values = best.to_numpy()
    image = axis.imshow(values, cmap=SEQUENTIAL_CMAP, vmin=0.0, vmax=1.0, aspect="equal")

    axis.set_xticks(range(len(languages)), languages)
    axis.set_yticks(range(len(languages)), languages)
    axis.set_xlabel("target language")
    axis.set_ylabel("source language")
    axis.set_title(title, color=INK_PRIMARY, loc="left", fontsize=11, pad=12)
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(False)

    for row_index in range(len(languages)):
        for column_index in range(len(languages)):
            value = values[row_index, column_index]
            if pd.isna(value):
                continue
            ink = "#ffffff" if value > 0.6 else INK_PRIMARY
            axis.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", color=ink, fontsize=9)

    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(metric.replace("_", " "), color=INK_SECONDARY)
    colorbar.outline.set_visible(False)
    save_figure(figure, path)


def plot_layer_curves(frame: pd.DataFrame, metric: str, languages: list[str], title: str, path: Path) -> None:
    colors = language_color_map(languages)
    sources = [language for language in languages if language in set(frame["source_language"])]
    columns = min(3, len(sources))
    rows = (len(sources) + columns - 1) // columns

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.4 * columns, 3.2 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for index, source in enumerate(sources):
        axis = axes[index // columns][index % columns]
        source_frame = frame[frame["source_language"] == source]
        for target in languages:
            target_frame = source_frame[source_frame["target_language"] == target].sort_values("layer")
            if target_frame.empty:
                continue
            in_language = target == source
            axis.plot(
                target_frame["layer"],
                target_frame[metric],
                color=colors[target],
                linewidth=2.4 if in_language else 1.6,
                linestyle="-" if in_language else (0, (4, 2)),
                label=target,
            )
        axis.set_title(f"source: {source}", fontsize=10, color=INK_SECONDARY, loc="left")
        axis.set_ylim(0.0, 1.0)
        axis.tick_params(labelsize=8)
        for spine_name in ("top", "right"):
            axis.spines[spine_name].set_visible(False)

    for index in range(len(sources), rows * columns):
        axes[index // columns][index % columns].set_visible(False)

    for axis in axes[-1]:
        axis.set_xlabel("layer", fontsize=9)
    for row in axes:
        row[0].set_ylabel(metric.replace("_", " "), fontsize=9)

    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        title="target",
        loc="upper left",
        bbox_to_anchor=(1.0, 0.94),
        frameon=False,
        fontsize=9,
        title_fontsize=9,
    )
    figure.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK_PRIMARY)
    figure.tight_layout(rect=(0, 0, 0.99, 0.95))
    save_figure(figure, path)


def plot_top_pairs(frame: pd.DataFrame, languages: list[str], top_k: int, title: str, path: Path) -> None:
    cross = frame[frame["source_language"] != frame["target_language"]]
    if cross.empty:
        print(f"[plot] no cross-lingual rows; skipping {path.name}")
        return

    best = (
        cross.groupby(["source_language", "target_language"])
        .agg(accuracy=("accuracy", "max"), macro_f1=("macro_f1", "max"))
        .reset_index()
        .sort_values("accuracy", ascending=False)
        .head(top_k)
    )

    colors = language_color_map(languages)
    labels = [f"{row.source_language} → {row.target_language}" for row in best.itertuples()]
    positions = range(len(best))

    figure, axis = plt.subplots(figsize=(7.2, 0.42 * len(best) + 1.6))
    axis.barh(
        positions,
        best["accuracy"],
        height=0.62,
        color=[colors[source] for source in best["source_language"]],
        edgecolor=SURFACE,
        linewidth=2,
    )
    for position, row in zip(positions, best.itertuples()):
        axis.text(
            row.accuracy + 0.012,
            position,
            f"{row.accuracy:.2f} (F1 {row.macro_f1:.2f})",
            va="center",
            fontsize=8.5,
            color=INK_SECONDARY,
        )

    axis.set_yticks(list(positions), labels)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("best accuracy across layers")
    axis.grid(axis="y", visible=False)
    axis.set_title(f"{title} (bar color = source language)", loc="left", fontsize=11, pad=12)
    for spine_name in ("top", "right", "left"):
        axis.spines[spine_name].set_visible(False)
    save_figure(figure, path)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if args.probes_dir is not None:
        summarize_probe_outputs(Path(args.probes_dir), input_dir)

    frame = load_summary_frames(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_aliases = sorted(frame["model_alias"].unique())
    multiple_models = len(model_aliases) > 1

    for model_alias in model_aliases:
        model_frame = frame[frame["model_alias"] == model_alias]
        languages = language_order(model_frame)

        plot_transfer_heatmap(
            model_frame,
            metric="accuracy",
            languages=languages,
            title=f"{model_alias}: best POS probe accuracy (max over layers)",
            path=figure_path(output_dir, model_alias, "best_accuracy_transfer_heatmap.svg", multiple_models),
        )
        plot_transfer_heatmap(
            model_frame,
            metric="macro_f1",
            languages=languages,
            title=f"{model_alias}: best POS probe macro F1 (max over layers)",
            path=figure_path(output_dir, model_alias, "best_macro_f1_transfer_heatmap.svg", multiple_models),
        )
        plot_layer_curves(
            model_frame,
            metric="accuracy",
            languages=languages,
            title=f"{model_alias}: probe accuracy by layer",
            path=figure_path(output_dir, model_alias, "layer_accuracy_curve.svg", multiple_models),
        )
        plot_layer_curves(
            model_frame,
            metric="macro_f1",
            languages=languages,
            title=f"{model_alias}: probe macro F1 by layer",
            path=figure_path(output_dir, model_alias, "layer_macro_f1_curve.svg", multiple_models),
        )
        plot_top_pairs(
            model_frame,
            languages=languages,
            top_k=args.top_k,
            title=f"{model_alias}: top cross-lingual transfer pairs",
            path=figure_path(output_dir, model_alias, "top_cross_lingual_pairs.svg", multiple_models),
        )

    print(f"[plot] figures written to {output_dir}")


if __name__ == "__main__":
    main()
