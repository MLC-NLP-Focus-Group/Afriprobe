from __future__ import annotations

import argparse
import os
from pathlib import Path

import modal


APP_NAME = "afriprobe-plasticity"
VOLUME_NAME = "afriprobe-plasticity"
HF_CACHE_VOLUME_NAME = "afriprobe-hf-cache"
OUTPUT_MOUNT = "/outputs"
HF_CACHE_MOUNT = "/cache/huggingface"
PROJECT_ROOT = Path(__file__).resolve().parent


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "tqdm",
        "sentencepiece",
        "protobuf",
    )
    .env(
        {
            "HF_HOME": HF_CACHE_MOUNT,
            "HUGGINGFACE_HUB_CACHE": f"{HF_CACHE_MOUNT}/hub",
            "TRANSFORMERS_CACHE": HF_CACHE_MOUNT,
            "HF_DATASETS_CACHE": f"{HF_CACHE_MOUNT}/datasets",
        }
    )
    .add_local_dir(
        PROJECT_ROOT / "analysis",
        remote_path="/root/analysis",
        ignore=["__pycache__", "*.pyc"],
    )
)

output_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]}) if os.environ.get("HF_TOKEN") else modal.Secret.from_dict({})
app = modal.App(APP_NAME, image=image)


@app.function(
    gpu=["L40S", "A10"],
    volumes={
        OUTPUT_MOUNT: output_volume,
        HF_CACHE_MOUNT: hf_cache_volume,
    },
    secrets=[hf_secret],
    timeout=60 * 60 * 8,
)
def run_plasticity_remote(
    before_model_name_or_path: str,
    after_model_name_or_path: str,
    before_alias: str,
    after_alias: str,
    tokenizer_name_or_path: str,
    before_tokenizer_name_or_path: str,
    after_tokenizer_name_or_path: str,
    languages_csv: str,
    split: str,
    batch_size: int,
    max_length: int,
    max_samples: int,
    representation_level: str,
    layers_csv: str,
    aggregation: str,
    torch_dtype: str,
    trust_remote_code: bool,
    skip_tokenizer_vocab_check: bool,
    dataset_name: str,
    text_field: str,
    streaming: bool,
    seed: int,
    shuffle_buffer_size: int,
    output_subdir: str,
) -> dict[str, str | int]:
    from analysis.plasticity import run_plasticity

    languages = [language.strip() for language in languages_csv.split(",") if language.strip()]
    layers = [layer.strip() for layer in layers_csv.split(",") if layer.strip()] or ["all"]
    tokenizer_name = tokenizer_name_or_path or before_model_name_or_path

    args = argparse.Namespace(
        before_model_name_or_path=before_model_name_or_path,
        after_model_name_or_path=after_model_name_or_path,
        before_alias=before_alias,
        after_alias=after_alias,
        tokenizer_name_or_path=tokenizer_name,
        before_tokenizer_name_or_path=before_tokenizer_name_or_path or before_model_name_or_path,
        after_tokenizer_name_or_path=after_tokenizer_name_or_path or after_model_name_or_path,
        skip_tokenizer_vocab_check=skip_tokenizer_vocab_check,
        languages=languages,
        split=split,
        batch_size=batch_size,
        max_length=max_length,
        max_samples=max_samples,
        max_tokens=None,
        representation_level=representation_level,
        layers=layers,
        layer_pairs=None,
        aggregation=aggregation,
        device="cuda",
        torch_dtype=torch_dtype,
        save_token_dtype="float16",
        trust_remote_code=trust_remote_code,
        dataset_name=dataset_name,
        text_field=text_field,
        streaming=streaming,
        seed=seed,
        shuffle_buffer_size=shuffle_buffer_size,
        output_dir=f"{OUTPUT_MOUNT}/{output_subdir}",
    )
    result = run_plasticity(args)
    output_volume.commit()
    hf_cache_volume.commit()
    return result


@app.local_entrypoint()
def main(
    before_model_name_or_path: str = "FacebookAI/xlm-roberta-large",
    after_model_name_or_path: str = "Davlan/afro-xlmr-large",
    before_alias: str = "xlmr-large",
    after_alias: str = "afro-xlmr-large",
    tokenizer_name_or_path: str = "FacebookAI/xlm-roberta-large",
    before_tokenizer_name_or_path: str = "FacebookAI/xlm-roberta-large",
    after_tokenizer_name_or_path: str = "Davlan/afro-xlmr-large",
    languages: str = "yor,ibo,hau,swa,wol,amh",
    split: str = "test",
    batch_size: int = 4,
    max_length: int = 256,
    max_samples: int = 1000,
    representation_level: str = "sentence",
    layers: str = "all",
    aggregation: str = "pooled",
    torch_dtype: str = "float16",
    trust_remote_code: bool = False,
    skip_tokenizer_vocab_check: bool = False,
    dataset_name: str = "google/fleurs",
    text_field: str = "transcription",
    streaming: bool = True,
    seed: int = 42,
    shuffle_buffer_size: int = 10_000,
    output_subdir: str = "plasticity",
) -> None:
    result = run_plasticity_remote.remote(
        before_model_name_or_path=before_model_name_or_path,
        after_model_name_or_path=after_model_name_or_path,
        before_alias=before_alias,
        after_alias=after_alias,
        tokenizer_name_or_path=tokenizer_name_or_path,
        before_tokenizer_name_or_path=before_tokenizer_name_or_path,
        after_tokenizer_name_or_path=after_tokenizer_name_or_path,
        languages_csv=languages,
        split=split,
        batch_size=batch_size,
        max_length=max_length,
        max_samples=max_samples,
        representation_level=representation_level,
        layers_csv=layers,
        aggregation=aggregation,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        skip_tokenizer_vocab_check=skip_tokenizer_vocab_check,
        dataset_name=dataset_name,
        text_field=text_field,
        streaming=streaming,
        seed=seed,
        shuffle_buffer_size=shuffle_buffer_size,
        output_subdir=output_subdir,
    )
    print(result)
    print(f"Outputs are in Modal Volume '{VOLUME_NAME}' under /{output_subdir}.")
