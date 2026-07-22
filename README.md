# AfriProbe

AfriProbe is a multilingual representation probing pipeline for studying cross-lingual transfer in low-resource African languages. The main experiment freezes a multilingual encoder, extracts token-level hidden states for MasakhaPOS, trains lightweight linear POS probes on one source language, and evaluates those probes on other target languages.

The current experiment setup uses:

| Code | Language |
| --- | --- |
| `yor` | Yoruba |
| `ibo` | Igbo |
| `hau` | Hausa |
| `swa` | Swahili |
| `wol` | Wolof |

## Repository Layout

```text
datasets/masakhapos.py          # MasakhaPOS loader and tokenizer-label alignment
extract/hidden_states.py        # Frozen encoder hidden-state extraction
probes/train_probe.py           # Single-layer linear probe training/evaluation
probes/train_layer_sweep.py     # Train/evaluate probes across all layers
analysis/cka.py                 # Pairwise representation alignment with linear CKA
analysis/summarize_probe_results.py  # Convert layer_sweep_metrics.json files to CSV summaries
analysis/plot_probe_results.py       # SVG plots from summarized probe CSV files
```

## 1. Set Up Python

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch for your machine first.

For a CPU-only machine:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For a CUDA machine, use a CUDA PyTorch wheel matching your environment. For example:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Then install the rest of the dependencies:

```bash
python -m pip install transformers datasets tqdm scikit-learn pandas matplotlib seaborn
```

Check the install:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 2. RunPod / Large-Disk Setup

Hidden states and model caches can be large. On RunPod, keep everything under `/workspace` instead of the small root disk:

```bash
mkdir -p /workspace/afriprobe_cache/{hf,pip,tmp}
mkdir -p /workspace/afriprobe/hidden_states
mkdir -p /workspace/afriprobe/probes
mkdir -p /workspace/afriprobe/analysis

export HF_HOME=/workspace/afriprobe_cache/hf
export HUGGINGFACE_HUB_CACHE=/workspace/afriprobe_cache/hf/hub
export TRANSFORMERS_CACHE=/workspace/afriprobe_cache/hf
export HF_DATASETS_CACHE=/workspace/afriprobe_cache/hf/datasets
export PIP_CACHE_DIR=/workspace/afriprobe_cache/pip
export TMPDIR=/workspace/afriprobe_cache/tmp
```

Optional: add those exports to `~/.bashrc` so future terminals use `/workspace` automatically.

## 3. Download MasakhaPOS

From the repository root, clone the official MasakhaPOS data:

```bash
git clone https://github.com/masakhane-io/masakhane-pos.git datasets/masakhane-pos
```

Confirm the data exists:

```bash
ls datasets/masakhane-pos/data
```

The commands below assume:

```bash
--data_dir datasets/masakhane-pos/data
```

If you cloned the dataset elsewhere, update `--data_dir` accordingly.

## 4. Extract Hidden States

This stage runs the frozen encoder once and saves hidden states to disk. The expensive encoder forward pass is done here; probe training later is much cheaper.

### XLM-R Base

```bash
python extract/hidden_states.py \
  --model_name_or_path FacebookAI/xlm-roberta-base \
  --model_alias xlmr \
  --languages yor ibo hau swa wol \
  --splits train validation test \
  --batch_size 8 \
  --max_length 256 \
  --output_dir /workspace/afriprobe/hidden_states \
  --device cuda \
  --data_dir datasets/masakhane-pos/data
```

Use `--device cpu` if you are not on a GPU machine.

XLM-R base has 13 hidden-state layers total: the embedding layer plus 12 transformer layers.

### AfroXLMR Large

AfroXLMR large has 25 hidden-state layers total: the embedding layer plus 24 transformer layers. The files are much larger than XLM-R base, so use a smaller extraction batch size:

```bash
python extract/hidden_states.py \
  --model_name_or_path Davlan/afro-xlmr-large \
  --model_alias afro-xlmr-large \
  --languages yor ibo hau swa wol \
  --splits train validation test \
  --batch_size 2 \
  --max_length 256 \
  --output_dir /workspace/afriprobe/hidden_states \
  --device cuda \
  --data_dir datasets/masakhane-pos/data
```

If storage is tight, extract only selected layers:

```bash
python extract/hidden_states.py \
  --model_name_or_path Davlan/afro-xlmr-large \
  --model_alias afro-xlmr-large \
  --languages yor ibo hau swa wol \
  --splits train validation test \
  --batch_size 2 \
  --max_length 256 \
  --output_dir /workspace/afriprobe/hidden_states \
  --device cuda \
  --data_dir datasets/masakhane-pos/data \
  --layers 4 8 12 16 20 24
```

Extraction output layout:

```text
/workspace/afriprobe/hidden_states/
└── xlmr/
    └── yor/
        ├── train/
        ├── validation/
        └── test/
```

Each split directory contains `chunk_*.pt` files and a `manifest.json`.

## 5. Train Linear Probes

The layer sweep trains one independent linear POS probe per saved layer for a source language. Each trained probe is evaluated on all target languages.

### One Source Language

Example: train Yoruba probes and evaluate on all five languages:

```bash
python probes/train_layer_sweep.py \
  --hidden_dir /workspace/afriprobe/hidden_states \
  --model_alias xlmr \
  --source_language yor \
  --target_languages yor ibo hau swa wol \
  --layers all \
  --train_split train \
  --eval_split test \
  --batch_size 8192 \
  --lr 1e-3 \
  --epochs 30 \
  --output_dir /workspace/afriprobe/probes \
  --device cuda
```

### Full XLM-R Transfer Matrix

```bash
for SRC in yor ibo hau swa wol; do
  python probes/train_layer_sweep.py \
    --hidden_dir /workspace/afriprobe/hidden_states \
    --model_alias xlmr \
    --source_language "$SRC" \
    --target_languages yor ibo hau swa wol \
    --layers all \
    --train_split train \
    --eval_split test \
    --batch_size 8192 \
    --lr 1e-3 \
    --epochs 30 \
    --output_dir /workspace/afriprobe/probes \
    --device cuda
done
```

### Full AfroXLMR Transfer Matrix

```bash
for SRC in yor ibo hau swa wol; do
  python probes/train_layer_sweep.py \
    --hidden_dir /workspace/afriprobe/hidden_states \
    --model_alias afro-xlmr-large \
    --source_language "$SRC" \
    --target_languages yor ibo hau swa wol \
    --layers all \
    --train_split train \
    --eval_split test \
    --batch_size 8192 \
    --lr 1e-3 \
    --epochs 30 \
    --output_dir /workspace/afriprobe/probes \
    --device cuda
done
```

Probe outputs are saved under:

```text
/workspace/afriprobe/probes/{model_alias}/source_{language}/
├── layer_0/
│   ├── probe.pt
│   └── metrics.json
└── layer_sweep_metrics.json
```

### Faster Probe Training With Token Cache

For large models, loading `.pt` chunks repeatedly can become the bottleneck. Use `--cache_dir` to build reusable flattened token matrices:

```bash
for SRC in yor ibo hau swa wol; do
  python probes/train_layer_sweep.py \
    --hidden_dir /workspace/afriprobe/hidden_states \
    --model_alias afro-xlmr-large \
    --source_language "$SRC" \
    --target_languages yor ibo hau swa wol \
    --layers all \
    --train_split train \
    --eval_split test \
    --batch_size 8192 \
    --lr 1e-3 \
    --epochs 30 \
    --output_dir /workspace/afriprobe/probes \
    --cache_dir /workspace/afriprobe/probe_token_cache \
    --device cuda
done
```

If `/dev/shm` has enough space, it is faster but temporary:

```bash
--cache_dir /dev/shm/afriprobe_probe_cache
```

## 6. Compute CKA Representation Alignment

CKA measures whether two languages have geometrically similar hidden-state spaces at the same layer. The script uses only valid POS-token positions where labels are not `-100`.

### XLM-R CKA

```bash
python analysis/cka.py \
  --hidden_dir /workspace/afriprobe/hidden_states \
  --model_alias xlmr \
  --languages yor ibo hau swa wol \
  --layers all \
  --split test \
  --max_tokens 5000 \
  --output_dir /workspace/afriprobe/analysis/cka \
  --device cuda
```

### AfroXLMR CKA

```bash
python analysis/cka.py \
  --hidden_dir /workspace/afriprobe/hidden_states \
  --model_alias afro-xlmr-large \
  --languages yor ibo hau swa wol \
  --layers all \
  --split test \
  --max_tokens 5000 \
  --output_dir /workspace/afriprobe/analysis/cka \
  --device cuda
```

CKA outputs are saved under:

```text
/workspace/afriprobe/analysis/cka/{model_alias}/test/
├── layer_0/cka_matrix.csv
├── layer_0/cka_matrix.json
└── cka_summary.json
```

Use `--device cpu` if GPU memory is limited. CKA can also be run with fewer tokens, for example `--max_tokens 2000`, for a faster pilot run.

## 7. Summarize Probe Results

After probe training, each source language has a `layer_sweep_metrics.json` file. Convert those JSON files into CSV tables with:

```bash
python analysis/summarize_probe_results.py \
  --input_dir /workspace/afriprobe/probes/xlmr \
  --output_dir analysis/probe_results \
  --model_alias xlmr \
  --languages yor ibo hau swa wol
```

For AfroXLMR:

```bash
python analysis/summarize_probe_results.py \
  --input_dir /workspace/afriprobe/probes/afro-xlmr-large \
  --output_dir analysis/probe_results_afro_xlmr_large \
  --model_alias afro-xlmr-large \
  --languages yor ibo hau swa wol
```

Expected CSV outputs:

```text
probe_transfer_long.csv
best_layer_by_pair.csv
best_accuracy_transfer_matrix.csv
best_macro_f1_transfer_matrix.csv
layer_average_summary.csv
source_summary.csv
target_summary.csv
```

## 8. Generate Probe Plots

If you have summarized probe CSVs in `analysis/probe_results`, generate SVG figures with:

```bash
python analysis/plot_probe_results.py \
  --input_dir analysis/probe_results \
  --output_dir analysis/probe_results/figures \
  --top_k 12
```

Expected figures include:

```text
best_accuracy_transfer_heatmap.svg
best_macro_f1_transfer_heatmap.svg
layer_accuracy_curve.svg
layer_macro_f1_curve.svg
top_cross_lingual_pairs.svg
```

## 9. Suggested Experiment Workflow

Recommended order:

1. Download MasakhaPOS.
2. Extract hidden states for all languages and splits.
3. Train layer-wise probes for each source language.
4. Summarize `layer_sweep_metrics.json` files into CSVs.
5. Generate probe plots.
6. Compute CKA for each language pair and layer.
7. Compare CKA scores against cross-lingual probe accuracy or macro F1.

For one model and five languages:

```text
5 source languages × all layers = trained probes
each probe evaluated on 5 target languages = source-target transfer matrix
```

## 10. Troubleshooting

### `No module named 'datasets'`

Install dependencies into the active Python environment:

```bash
which python
python -m pip install datasets transformers tqdm
```

### Hugging Face downloads fill `/root/.cache`

Set cache variables before running extraction:

```bash
export HF_HOME=/workspace/afriprobe_cache/hf
export HUGGINGFACE_HUB_CACHE=/workspace/afriprobe_cache/hf/hub
export TRANSFORMERS_CACHE=/workspace/afriprobe_cache/hf
export HF_DATASETS_CACHE=/workspace/afriprobe_cache/hf/datasets
```

### `torch.save` or download fails with `No space left on device`

Check disk usage:

```bash
df -h
```

Make sure `--output_dir`, Hugging Face cache, pip cache, and temporary files point to `/workspace`.

### CUDA out of memory during extraction

Lower extraction batch size:

```bash
--batch_size 1
```

This is especially useful for `Davlan/afro-xlmr-large`.

### Probe training is slow

Use a larger probe batch size and a cache directory:

```bash
--batch_size 8192 --cache_dir /workspace/afriprobe/probe_token_cache
```

Probe training can use larger batches than hidden-state extraction because the encoder is not being run.
