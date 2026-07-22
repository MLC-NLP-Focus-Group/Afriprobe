from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from datasets import Audio, load_dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


FLEURS_LANGUAGE_CONFIGS = {
    "yoruba": "yo_ng",
    "yor": "yo_ng",
    "yo": "yo_ng",
    "yo_ng": "yo_ng",
    "igbo": "ig_ng",
    "ibo": "ig_ng",
    "ig": "ig_ng",
    "ig_ng": "ig_ng",
    "hausa": "ha_ng",
    "hau": "ha_ng",
    "ha": "ha_ng",
    "ha_ng": "ha_ng",
    "swahili": "sw_ke",
    "swa": "sw_ke",
    "swh": "sw_ke",
    "sw": "sw_ke",
    "sw_ke": "sw_ke",
    "wolof": "wo_sn",
    "wol": "wo_sn",
    "wo": "wo_sn",
    "wo_sn": "wo_sn",
    "amharic": "am_et",
    "amh": "am_et",
    "am": "am_et",
    "am_et": "am_et",
    "english": "en_us",
    "eng": "en_us",
    "en": "en_us",
    "en_us": "en_us",
    "french": "fr_fr",
    "fra": "fr_fr",
    "fr": "fr_fr",
    "fr_fr": "fr_fr",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure representation plasticity between a base multilingual model and "
            "its continued-adapted counterpart using layer-wise linear CKA."
        )
    )
    parser.add_argument("--before_model_name_or_path", required=True)
    parser.add_argument("--after_model_name_or_path", required=True)
    parser.add_argument("--before_alias", required=True)
    parser.add_argument("--after_alias", required=True)
    parser.add_argument("--tokenizer_name_or_path", default=None)
    parser.add_argument("--before_tokenizer_name_or_path", default=None)
    parser.add_argument("--after_tokenizer_name_or_path", default=None)
    parser.add_argument("--skip_tokenizer_vocab_check", action="store_true")
    parser.add_argument("--languages", nargs="+", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--max_tokens", type=int, default=None, help="Deprecated alias for --max_samples.")
    parser.add_argument("--representation_level", choices=["sentence", "token"], default="sentence")
    parser.add_argument("--layers", nargs="+", default=["all"])
    parser.add_argument("--layer_pairs", nargs="+", default=None, help="Optional explicit pairs like 0:0 1:1.")
    parser.add_argument("--aggregation", choices=["pooled", "per_language", "both"], default="pooled")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch_dtype", choices=["float16", "float32", "bfloat16"], default="float16")
    parser.add_argument("--save_token_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--dataset_name", default="google/fleurs")
    parser.add_argument("--text_field", default="transcription")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle_buffer_size", type=int, default=10_000)
    parser.add_argument("--output_dir", default="outputs/analysis/plasticity")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_arg)


def resolve_model_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "float32" or device.type == "cpu":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported torch dtype: {dtype_arg}")


def resolve_save_dtype(dtype_arg: str) -> torch.dtype:
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "float32":
        return torch.float32
    raise ValueError(f"Unsupported save dtype: {dtype_arg}")


def parse_int_list(items: list[str]) -> list[int]:
    values = []
    for item in items:
        for part in item.split(","):
            part = part.strip()
            if part:
                values.append(int(part))
    return values


def parse_layer_pairs(
    layer_args: list[str],
    layer_pair_args: list[str] | None,
    num_before_layers: int,
    num_after_layers: int,
) -> list[tuple[int, int]]:
    if layer_pair_args:
        pairs = []
        for pair_arg in layer_pair_args:
            if ":" not in pair_arg:
                raise ValueError(f"Layer pair must look like before:after, got {pair_arg!r}")
            before_layer, after_layer = pair_arg.split(":", maxsplit=1)
            pairs.append((int(before_layer), int(after_layer)))
    elif layer_args == ["all"]:
        pairs = [(layer, layer) for layer in range(min(num_before_layers, num_after_layers))]
    else:
        layers = parse_int_list(layer_args)
        pairs = [(layer, layer) for layer in layers]

    for before_layer, after_layer in pairs:
        if before_layer < 0 or before_layer >= num_before_layers:
            raise ValueError(f"Before layer {before_layer} is outside 0..{num_before_layers - 1}.")
        if after_layer < 0 or after_layer >= num_after_layers:
            raise ValueError(f"After layer {after_layer} is outside 0..{num_after_layers - 1}.")
    return pairs


def resolve_fleurs_config(language: str) -> str:
    key = language.strip().lower().replace("-", "_")
    return FLEURS_LANGUAGE_CONFIGS.get(key, key)


def batched(iterable: Any, batch_size: int) -> Any:
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def load_fleurs_split(dataset_name: str, language: str, split: str, streaming: bool) -> Any:
    config_name = resolve_fleurs_config(language)
    print(f"[plasticity] loading FLEURS config: language={language} config={config_name} split={split}")
    dataset = load_dataset(dataset_name, config_name, split=split, streaming=streaming)
    if "audio" in getattr(dataset, "features", {}):
        dataset = dataset.cast_column("audio", Audio(decode=False))
    return dataset


def extract_text(example: dict[str, Any], text_field: str) -> str:
    if text_field in example and example[text_field] is not None:
        return str(example[text_field]).strip()
    if text_field != "raw_transcription" and "raw_transcription" in example and example["raw_transcription"] is not None:
        return str(example["raw_transcription"]).strip()
    if text_field != "transcription" and "transcription" in example and example["transcription"] is not None:
        return str(example["transcription"]).strip()
    available = ", ".join(example.keys())
    raise KeyError(f"Could not find text field {text_field!r}; available fields: {available}")


def example_id(example: dict[str, Any], language: str, index: int, text: str) -> str:
    for key in ("id", "path", "audio", "file"):
        if key not in example or example[key] is None:
            continue
        value = example[key]
        if isinstance(value, dict) and "path" in value and value["path"] is not None:
            return str(value["path"])
        if not isinstance(value, dict):
            return str(value)
    digest = hashlib.sha1(f"{language}\n{index}\n{text}".encode("utf-8")).hexdigest()[:16]
    return f"{language}:{index}:{digest}"


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def append_limited(parts: list[torch.Tensor], values: torch.Tensor, max_samples: int) -> None:
    if values.numel() == 0:
        return
    current = sum(part.shape[0] for part in parts)
    remaining = max_samples - current
    if remaining <= 0:
        return
    parts.append(values[:remaining].detach().cpu())


def enough_samples(buffers: dict[str, dict[tuple[int, int], dict[str, list[torch.Tensor]]]], max_samples: int) -> bool:
    for layer_buffers in buffers.values():
        for pair_buffer in layer_buffers.values():
            count = sum(part.shape[0] for part in pair_buffer["before"])
            if count < max_samples:
                return False
    return True


def mean_pool_tokens(hidden: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    mask = valid_mask.to(device=hidden.device).unsqueeze(-1).to(dtype=hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return summed / counts


def load_models(
    before_model_name_or_path: str,
    after_model_name_or_path: str,
    model_dtype: torch.dtype,
    device: torch.device,
    trust_remote_code: bool,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    before_model = AutoModel.from_pretrained(
        before_model_name_or_path,
        torch_dtype=model_dtype,
        trust_remote_code=trust_remote_code,
    )
    after_model = AutoModel.from_pretrained(
        after_model_name_or_path,
        torch_dtype=model_dtype,
        trust_remote_code=trust_remote_code,
    )
    before_model.requires_grad_(False).eval().to(device)
    after_model.requires_grad_(False).eval().to(device)
    return before_model, after_model


def ensure_padding_token(tokenizer: Any) -> None:
    if tokenizer.pad_token is not None:
        return
    if tokenizer.eos_token is None:
        raise ValueError(
            "Tokenizer has no pad token or EOS token. Set a pad token before running plasticity analysis."
        )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(f"[plasticity] tokenizer had no pad token; using eos token as pad: {tokenizer.pad_token!r}")


def validate_tokenizer_compatibility(
    tokenizer: Any,
    before_tokenizer: Any,
    after_tokenizer: Any,
    before_model: torch.nn.Module,
    after_model: torch.nn.Module,
    skip_vocab_check: bool,
) -> None:
    before_vocab_size = before_model.get_input_embeddings().num_embeddings
    after_vocab_size = after_model.get_input_embeddings().num_embeddings
    tokenizer_size = len(tokenizer)

    if tokenizer_size > before_vocab_size:
        raise ValueError(
            f"Tokenizer size {tokenizer_size} exceeds before-model embedding vocabulary {before_vocab_size}."
        )
    if tokenizer_size > after_vocab_size:
        raise ValueError(
            f"Tokenizer size {tokenizer_size} exceeds after-model embedding vocabulary {after_vocab_size}."
        )

    if skip_vocab_check:
        print("[plasticity] skipping tokenizer vocabulary equality check")
        return

    if before_tokenizer.get_vocab() != after_tokenizer.get_vocab():
        raise ValueError(
            "The before and after models do not use identical tokenizer vocabularies. "
            "Before/after CKA with shared input_ids is not directly aligned. "
            "Use a true continued-pretraining pair with the same tokenizer, or pass "
            "--skip_tokenizer_vocab_check only for exploratory analysis."
        )
    if tokenizer.get_vocab() != before_tokenizer.get_vocab():
        raise ValueError(
            "The active tokenizer does not match the before-model tokenizer vocabulary. "
            "Use --tokenizer_name_or_path/--before_tokenizer_name_or_path consistently."
        )

    print(
        f"[plasticity] tokenizer compatibility ok: tokenizer_size={tokenizer_size}, "
        f"before_embeddings={before_vocab_size}, after_embeddings={after_vocab_size}"
    )


def linear_cka(left: torch.Tensor, right: torch.Tensor, device: torch.device) -> float:
    left = left.to(device=device, dtype=torch.float32)
    right = right.to(device=device, dtype=torch.float32)
    left = left - left.mean(dim=0, keepdim=True)
    right = right - right.mean(dim=0, keepdim=True)

    hsic = torch.linalg.matrix_norm(left.T @ right, ord="fro") ** 2
    left_norm = torch.linalg.matrix_norm(left.T @ left, ord="fro")
    right_norm = torch.linalg.matrix_norm(right.T @ right, ord="fro")
    return float((hsic / (left_norm * right_norm + 1e-8)).detach().cpu())


def init_buffers(
    groups: list[str],
    layer_pairs: list[tuple[int, int]],
) -> dict[str, dict[tuple[int, int], dict[str, list[torch.Tensor]]]]:
    return {
        group: {
            pair: {"before": [], "after": []}
            for pair in layer_pairs
        }
        for group in groups
    }


def combine_language_buffers(
    buffers: dict[str, dict[tuple[int, int], dict[str, list[torch.Tensor]]]],
    languages: list[str],
    layer_pairs: list[tuple[int, int]],
) -> dict[tuple[int, int], dict[str, list[torch.Tensor]]]:
    combined = {pair: {"before": [], "after": []} for pair in layer_pairs}
    for language in languages:
        for pair in layer_pairs:
            combined[pair]["before"].extend(buffers[language][pair]["before"])
            combined[pair]["after"].extend(buffers[language][pair]["after"])
    return combined


def collect_representations(
    before_model: torch.nn.Module,
    after_model: torch.nn.Module,
    tokenizer: Any,
    languages: list[str],
    split: str,
    batch_size: int,
    max_length: int,
    dataset_name: str,
    text_field: str,
    streaming: bool,
    seed: int,
    shuffle_buffer_size: int,
    layer_pairs: list[tuple[int, int]],
    aggregation: str,
    representation_level: str,
    max_samples: int,
    save_dtype: torch.dtype,
    device: torch.device,
) -> tuple[
    dict[str, dict[tuple[int, int], dict[str, list[torch.Tensor]]]],
    dict[str, list[dict[str, Any]]],
]:
    buffers = init_buffers(languages, layer_pairs)
    sampled_examples: dict[str, list[dict[str, Any]]] = {language: [] for language in languages}

    with torch.inference_mode():
        for language_index, language in enumerate(languages):
            dataset = load_fleurs_split(
                dataset_name=dataset_name,
                language=language,
                split=split,
                streaming=streaming,
            )
            if streaming and shuffle_buffer_size > 0:
                dataset = dataset.shuffle(seed=seed + language_index, buffer_size=shuffle_buffer_size)
            elif not streaming:
                dataset = dataset.shuffle(seed=seed + language_index)

            seen_count = 0
            for examples in tqdm(batched(dataset, batch_size), desc=f"plasticity {language}/{split}", leave=False):
                batch_texts = []
                batch_metadata = []
                for example in examples:
                    text = extract_text(example, text_field)
                    if not text:
                        seen_count += 1
                        continue
                    batch_texts.append(text)
                    batch_metadata.append(
                        {
                            "language": language,
                            "fleurs_config": resolve_fleurs_config(language),
                            "source_index_after_shuffle": seen_count,
                            "example_id": example_id(example, language, seen_count, text),
                            "text_sha1": text_hash(text),
                            "text_preview": text[:160],
                        }
                    )
                    seen_count += 1

                texts = batch_texts
                if not texts:
                    continue
                encoding = tokenizer(
                    texts,
                    truncation=True,
                    padding=True,
                    max_length=max_length,
                    return_tensors="pt",
                    return_special_tokens_mask=True,
                )
                input_ids = encoding["input_ids"].to(device)
                attention_mask = encoding["attention_mask"].to(device)
                special_tokens_mask = encoding["special_tokens_mask"].bool()
                valid_positions = encoding["attention_mask"].bool().cpu() & ~special_tokens_mask.cpu()

                before_outputs = before_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                after_outputs = after_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

                for before_layer, after_layer in layer_pairs:
                    before_hidden = before_outputs.hidden_states[before_layer]
                    after_hidden = after_outputs.hidden_states[after_layer]
                    if representation_level == "sentence":
                        before_values = mean_pool_tokens(before_hidden, valid_positions).to(dtype=save_dtype)
                        after_values = mean_pool_tokens(after_hidden, valid_positions).to(dtype=save_dtype)
                    else:
                        before_values = before_hidden.detach().cpu()[valid_positions].to(dtype=save_dtype)
                        after_values = after_hidden.detach().cpu()[valid_positions].to(dtype=save_dtype)

                    append_limited(buffers[language][(before_layer, after_layer)]["before"], before_values, max_samples)
                    append_limited(buffers[language][(before_layer, after_layer)]["after"], after_values, max_samples)

                remaining_metadata = max_samples - len(sampled_examples[language])
                if remaining_metadata > 0:
                    sampled_examples[language].extend(batch_metadata[:remaining_metadata])

                if enough_samples({language: buffers[language]}, max_samples):
                    break

            print(f"[plasticity] collected language={language}")

    if aggregation == "pooled":
        return {"ALL": combine_language_buffers(buffers, languages, layer_pairs)}, sampled_examples
    if aggregation == "both":
        return {"ALL": combine_language_buffers(buffers, languages, layer_pairs), **buffers}, sampled_examples
    return buffers, sampled_examples


def summarize_buffers(
    buffers: dict[str, dict[tuple[int, int], dict[str, list[torch.Tensor]]]],
    before_alias: str,
    after_alias: str,
    before_model_name_or_path: str,
    after_model_name_or_path: str,
    split: str,
    representation_level: str,
    max_samples: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows = []
    for group, layer_buffers in buffers.items():
        for (before_layer, after_layer), pair_buffer in sorted(layer_buffers.items()):
            if not pair_buffer["before"] or not pair_buffer["after"]:
                continue
            before_matrix = torch.cat(pair_buffer["before"], dim=0)
            after_matrix = torch.cat(pair_buffer["after"], dim=0)
            n_samples = min(before_matrix.shape[0], after_matrix.shape[0], max_samples)
            before_matrix = before_matrix[:n_samples]
            after_matrix = after_matrix[:n_samples]
            cka = linear_cka(before_matrix, after_matrix, device=device)
            rows.append(
                {
                    "comparison": f"{before_alias}_to_{after_alias}",
                    "before_alias": before_alias,
                    "after_alias": after_alias,
                    "before_model_name_or_path": before_model_name_or_path,
                    "after_model_name_or_path": after_model_name_or_path,
                    "language_group": group,
                    "split": split,
                    "layer": before_layer if before_layer == after_layer else f"{before_layer}:{after_layer}",
                    "before_layer": before_layer,
                    "after_layer": after_layer,
                    "cka": cka,
                    "plasticity": 1.0 - cka,
                    "representation_level": representation_level,
                    "num_samples": n_samples,
                    "before_hidden_dim": before_matrix.shape[1],
                    "after_hidden_dim": after_matrix.shape[1],
                }
            )
            print(
                f"[plasticity] group={group} layer={before_layer}->{after_layer} "
                f"cka={cka:.4f} plasticity={1.0 - cka:.4f} "
                f"{representation_level}_samples={n_samples}"
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "comparison",
        "before_alias",
        "after_alias",
        "before_model_name_or_path",
        "after_model_name_or_path",
        "language_group",
        "split",
        "layer",
        "before_layer",
        "after_layer",
        "cka",
        "plasticity",
        "representation_level",
        "num_samples",
        "before_hidden_dim",
        "after_hidden_dim",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def scale(value: float, low: float, high: float, pixel_low: float, pixel_high: float) -> float:
    if high <= low:
        return (pixel_low + pixel_high) / 2
    return pixel_low + (value - low) * (pixel_high - pixel_low) / (high - low)


def write_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    pooled_rows = [row for row in rows if row["language_group"] == "ALL"] or rows
    pooled_rows = sorted(pooled_rows, key=lambda row: int(row["before_layer"]))
    layers = [int(row["before_layer"]) for row in pooled_rows]
    cka_values = [float(row["cka"]) for row in pooled_rows]
    plasticity_values = [float(row["plasticity"]) for row in pooled_rows]

    width, height = 920, 520
    left, right, top, bottom = 82, 42, 86, 72
    plot_width = width - left - right
    plot_height = height - top - bottom
    low = 0.0
    high = 1.0

    def point_path(values: list[float]) -> str:
        points = []
        for layer, value in zip(layers, values):
            x = scale(layer, min(layers), max(layers), left, left + plot_width)
            y = scale(value, low, high, top + plot_height, top)
            points.append((x, y))
        return " ".join(("M" if index == 0 else "L") + f" {x:.2f} {y:.2f}" for index, (x, y) in enumerate(points))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        'text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #1f2937; }',
        ".title { font-size: 24px; font-weight: 700; }",
        ".subtitle { font-size: 13px; fill: #6b7280; }",
        ".axis { font-size: 12px; fill: #4b5563; }",
        ".small { font-size: 11px; fill: #4b5563; }",
        ".grid { stroke: #e5e7eb; stroke-width: 1; }",
        ".axis-line { stroke: #9ca3af; stroke-width: 1.2; }",
        "</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        '<text class="title" x="32" y="42">Layer-wise Representation Plasticity</text>',
        '<text class="subtitle" x="32" y="66">CKA compares before vs. continued-adapted hidden states; lower CKA means larger change.</text>',
    ]

    for tick in range(0, 6):
        value = tick / 5
        y = scale(value, low, high, top + plot_height, top)
        parts.append(f'<line class="grid" x1="{left}" y1="{y}" x2="{width - right}" y2="{y}"/>')
        parts.append(f'<text class="axis" x="{left - 12}" y="{y + 4}" text-anchor="end">{value:.1f}</text>')

    parts.append(f'<line class="axis-line" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>')
    parts.append(f'<line class="axis-line" x1="{left}" y1="{top + plot_height}" x2="{width - right}" y2="{top + plot_height}"/>')

    for layer in layers:
        x = scale(layer, min(layers), max(layers), left, left + plot_width)
        parts.append(f'<text class="axis" x="{x}" y="{height - 38}" text-anchor="middle">{layer}</text>')

    parts.append(f'<text class="axis" x="{width / 2}" y="{height - 12}" text-anchor="middle">Layer</text>')
    parts.append(f'<text class="axis" x="18" y="{top + plot_height / 2}" transform="rotate(-90 18 {top + plot_height / 2})" text-anchor="middle">Score</text>')
    parts.append(f'<path d="{point_path(cka_values)}" fill="none" stroke="#2563eb" stroke-width="3"/>')
    parts.append(f'<path d="{point_path(plasticity_values)}" fill="none" stroke="#dc2626" stroke-width="3"/>')

    for layer, cka, plasticity in zip(layers, cka_values, plasticity_values):
        x = scale(layer, min(layers), max(layers), left, left + plot_width)
        y_cka = scale(cka, low, high, top + plot_height, top)
        y_plasticity = scale(plasticity, low, high, top + plot_height, top)
        parts.append(f'<circle cx="{x:.2f}" cy="{y_cka:.2f}" r="4" fill="#2563eb"/>')
        parts.append(f'<circle cx="{x:.2f}" cy="{y_plasticity:.2f}" r="4" fill="#dc2626"/>')

    legend_x = width - 210
    parts.append(f'<line x1="{legend_x}" y1="35" x2="{legend_x + 28}" y2="35" stroke="#2563eb" stroke-width="3"/>')
    parts.append(f'<text class="small" x="{legend_x + 36}" y="39">CKA similarity</text>')
    parts.append(f'<line x1="{legend_x}" y1="58" x2="{legend_x + 28}" y2="58" stroke="#dc2626" stroke-width="3"/>')
    parts.append(f'<text class="small" x="{legend_x + 36}" y="62">Plasticity = 1 - CKA</text>')
    parts.append("</svg>")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def run_plasticity(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    model_dtype = resolve_model_dtype(args.torch_dtype, device)
    save_dtype = resolve_save_dtype(args.save_token_dtype)
    before_tokenizer_name_or_path = args.before_tokenizer_name_or_path or args.before_model_name_or_path
    after_tokenizer_name_or_path = args.after_tokenizer_name_or_path or args.after_model_name_or_path
    tokenizer_name_or_path = args.tokenizer_name_or_path or before_tokenizer_name_or_path
    max_samples = args.max_tokens if args.max_tokens is not None else args.max_samples

    print(f"[plasticity] loading tokenizer: {tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if not tokenizer.is_fast:
        raise ValueError("Plasticity analysis requires a fast tokenizer for text-token alignment.")
    ensure_padding_token(tokenizer)
    before_tokenizer = AutoTokenizer.from_pretrained(
        before_tokenizer_name_or_path,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    after_tokenizer = AutoTokenizer.from_pretrained(
        after_tokenizer_name_or_path,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )

    print(
        f"[plasticity] loading models: before={args.before_model_name_or_path}, "
        f"after={args.after_model_name_or_path}, dtype={model_dtype}, device={device}"
    )
    before_model, after_model = load_models(
        before_model_name_or_path=args.before_model_name_or_path,
        after_model_name_or_path=args.after_model_name_or_path,
        model_dtype=model_dtype,
        device=device,
        trust_remote_code=args.trust_remote_code,
    )
    validate_tokenizer_compatibility(
        tokenizer=tokenizer,
        before_tokenizer=before_tokenizer,
        after_tokenizer=after_tokenizer,
        before_model=before_model,
        after_model=after_model,
        skip_vocab_check=args.skip_tokenizer_vocab_check,
    )

    num_before_layers = before_model.config.num_hidden_layers + 1
    num_after_layers = after_model.config.num_hidden_layers + 1
    layer_pairs = parse_layer_pairs(args.layers, args.layer_pairs, num_before_layers, num_after_layers)
    print(f"[plasticity] layer_pairs={layer_pairs}")

    buffers, sampled_examples = collect_representations(
        before_model=before_model,
        after_model=after_model,
        tokenizer=tokenizer,
        languages=args.languages,
        split=args.split,
        batch_size=args.batch_size,
        max_length=args.max_length,
        dataset_name=args.dataset_name,
        text_field=args.text_field,
        streaming=args.streaming,
        seed=args.seed,
        shuffle_buffer_size=args.shuffle_buffer_size,
        layer_pairs=layer_pairs,
        aggregation=args.aggregation,
        representation_level=args.representation_level,
        max_samples=max_samples,
        save_dtype=save_dtype,
        device=device,
    )
    rows = summarize_buffers(
        buffers=buffers,
        before_alias=args.before_alias,
        after_alias=args.after_alias,
        before_model_name_or_path=args.before_model_name_or_path,
        after_model_name_or_path=args.after_model_name_or_path,
        split=args.split,
        representation_level=args.representation_level,
        max_samples=max_samples,
        device=device,
    )
    if not rows:
        raise RuntimeError("No rows were produced. Check languages, split, and dataset paths.")

    run_name = f"{args.before_alias}_to_{args.after_alias}"
    output_dir = Path(args.output_dir) / run_name
    csv_path = output_dir / "plasticity_summary.csv"
    json_path = output_dir / "plasticity_summary.json"
    samples_path = output_dir / "sampled_examples.json"
    svg_path = output_dir / "plasticity_curve.svg"

    payload = {
        "before_model_name_or_path": args.before_model_name_or_path,
        "after_model_name_or_path": args.after_model_name_or_path,
        "before_alias": args.before_alias,
        "after_alias": args.after_alias,
        "tokenizer_name_or_path": tokenizer_name_or_path,
        "before_tokenizer_name_or_path": before_tokenizer_name_or_path,
        "after_tokenizer_name_or_path": after_tokenizer_name_or_path,
        "tokenizer_vocab_check": not args.skip_tokenizer_vocab_check,
        "dataset_name": args.dataset_name,
        "text_field": args.text_field,
        "streaming": args.streaming,
        "seed": args.seed,
        "shuffle_buffer_size": args.shuffle_buffer_size,
        "fleurs_configs": {language: resolve_fleurs_config(language) for language in args.languages},
        "languages": args.languages,
        "split": args.split,
        "representation_level": args.representation_level,
        "max_samples": max_samples,
        "layer_pairs": layer_pairs,
        "aggregation": args.aggregation,
        "results": rows,
    }
    write_csv(csv_path, rows)
    write_json(json_path, payload)
    write_json(samples_path, sampled_examples)
    write_svg(svg_path, rows)
    print(f"[plasticity] wrote {csv_path}")
    print(f"[plasticity] wrote {json_path}")
    print(f"[plasticity] wrote {samples_path}")
    print(f"[plasticity] wrote {svg_path}")
    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "samples": str(samples_path),
        "svg": str(svg_path),
        "num_rows": len(rows),
    }


def main() -> None:
    args = parse_args()
    run_plasticity(args)


if __name__ == "__main__":
    main()
