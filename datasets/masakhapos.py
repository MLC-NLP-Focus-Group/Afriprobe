from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import torch
from datasets import ClassLabel, Sequence, load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


IGNORE_INDEX = -100
DEFAULT_DATASET_NAME = "masakhapos"
DEFAULT_GITHUB_REPO = "https://raw.githubusercontent.com/masakhane-io/masakhane-pos"
TOKEN_FIELDS = ("tokens", "words", "word")
LABEL_FIELDS = ("labels", "pos_tags", "upos", "upos_tags", "tags")
UPOS_LABELS = [
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
]
SPLIT_ALIASES = {
    "train": ("train",),
    "validation": ("validation", "dev", "valid", "val"),
    "dev": ("dev", "validation", "valid", "val"),
    "test": ("test",),
}


@dataclass(frozen=True)
class LabelInfo:
    field_name: str
    label_list: list[str]
    label_to_id: dict[str, int] | None


class SimpleMasakhaPOSCorpus:
    def __init__(self, examples: list[dict[str, list[Any]]], label_list: list[str]) -> None:
        self.examples = examples
        self.features = {
            "tokens": None,
            "labels": Sequence(ClassLabel(names=label_list)),
        }

    def __iter__(self):
        return iter(self.examples)

    def __len__(self) -> int:
        return len(self.examples)


def _find_existing_field(features: Any, candidates: tuple[str, ...], kind: str) -> str:
    for field_name in candidates:
        if field_name in features:
            return field_name
    available = ", ".join(features.keys())
    raise ValueError(f"Could not find a {kind} field. Tried {candidates}; available fields: {available}")


def _class_label_from_feature(feature: Any) -> ClassLabel | None:
    if isinstance(feature, ClassLabel):
        return feature
    if isinstance(feature, Sequence) and isinstance(feature.feature, ClassLabel):
        return feature.feature
    if hasattr(feature, "feature") and isinstance(feature.feature, ClassLabel):
        return feature.feature
    return None


def _build_label_info(dataset: Any, label_field: str) -> LabelInfo:
    feature = dataset.features[label_field]
    class_label = _class_label_from_feature(feature)
    if class_label is not None:
        return LabelInfo(
            field_name=label_field,
            label_list=list(class_label.names),
            label_to_id=None,
        )

    observed: set[Any] = set()
    for example in dataset:
        observed.update(example[label_field])

    if all(isinstance(label, str) for label in observed):
        label_list = sorted(observed)
        return LabelInfo(
            field_name=label_field,
            label_list=label_list,
            label_to_id={label: index for index, label in enumerate(label_list)},
        )

    if all(isinstance(label, int) for label in observed):
        max_label = max(observed) if observed else -1
        return LabelInfo(
            field_name=label_field,
            label_list=[str(label) for label in range(max_label + 1)],
            label_to_id=None,
        )

    raise ValueError(f"Unsupported label values in field '{label_field}': {sorted(type(x).__name__ for x in observed)}")


def _normalise_split_names(split: str) -> tuple[str, ...]:
    return SPLIT_ALIASES.get(split, (split,))


def _github_candidate_urls(github_repo: str, language: str, split: str) -> list[str]:
    split_names = _normalise_split_names(split)
    extensions = ("txt", "tsv", "conll", "conllu")
    branches = ("main", "master")
    templates = (
        "data/{language}/{split}.{extension}",
        "data/{language}/{language}_{split}.{extension}",
        "data/{language}/{language}-{split}.{extension}",
        "data/{language}_{split}.{extension}",
        "data/{language}-{split}.{extension}",
        "data/{split}/{language}.{extension}",
        "data/{split}/{language}_{split}.{extension}",
    )

    urls = []
    for branch in branches:
        for split_name in split_names:
            for template in templates:
                for extension in extensions:
                    path = template.format(language=language, split=split_name, extension=extension)
                    urls.append(f"{github_repo.rstrip('/')}/{branch}/{path}")
    return urls


def _read_url(url: str) -> str:
    with urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8")


def _read_local_file(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        return handle.read()


def _parse_pos_text(text: str) -> tuple[list[dict[str, list[Any]]], list[str]]:
    examples = []
    tokens = []
    labels = []
    label_set = set()

    def flush_sentence() -> None:
        nonlocal tokens, labels
        if tokens:
            examples.append({"tokens": tokens, "labels": labels})
            tokens = []
            labels = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            flush_sentence()
            continue
        if line.startswith("#"):
            continue

        fields = line.split("\t") if "\t" in line else line.split()
        if len(fields) < 2:
            continue

        if len(fields) >= 4 and fields[0].replace(".", "", 1).replace("-", "", 1).isdigit():
            token_id = fields[0]
            if "-" in token_id or "." in token_id:
                continue
            token = fields[1]
            label = fields[3]
        else:
            token = fields[0]
            label = fields[-1]

        tokens.append(token)
        labels.append(label)
        label_set.add(label)

    flush_sentence()

    if not examples:
        raise ValueError("No POS-tagged sentences could be parsed from the dataset file.")

    if label_set.issubset(set(UPOS_LABELS)):
        label_list = UPOS_LABELS
    else:
        label_list = sorted(label_set)
    label_to_id = {label: index for index, label in enumerate(label_list)}
    for example in examples:
        example["labels"] = [label_to_id[label] for label in example["labels"]]

    return examples, label_list


def _load_github_dataset(github_repo: str, language: str, split: str) -> SimpleMasakhaPOSCorpus:
    errors = []
    for url in _github_candidate_urls(github_repo, language, split):
        try:
            text = _read_url(url)
            examples, label_list = _parse_pos_text(text)
            print(f"[dataset] loaded MasakhaPOS from {url}")
            return SimpleMasakhaPOSCorpus(examples=examples, label_list=label_list)
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            errors.append(f"{url}: {exc}")

    preview = "\n".join(errors[:8])
    raise RuntimeError(
        "Could not load MasakhaPOS from Hugging Face or the GitHub fallback. "
        "Tried raw GitHub candidate paths such as:\n"
        f"{preview}"
    )


def _load_local_dataset(data_dir: Path, language: str, split: str) -> SimpleMasakhaPOSCorpus:
    split_names = _normalise_split_names(split)
    candidates = []
    for split_name in split_names:
        candidates.extend(
            [
                data_dir / language / f"{split_name}.txt",
                data_dir / language / f"{split_name}.tsv",
                data_dir / language / f"{split_name}.conll",
                data_dir / language / f"{split_name}.conllu",
                data_dir / language / f"{language}_{split_name}.txt",
                data_dir / language / f"{language}_{split_name}.tsv",
                data_dir / f"{language}_{split_name}.txt",
                data_dir / f"{language}_{split_name}.tsv",
            ]
        )

    allowed_suffixes = {".txt", ".tsv", ".conll", ".conllu"}
    for path in data_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in allowed_suffixes:
            continue
        path_text = str(path).lower()
        if language.lower() in path_text and any(split_name.lower() in path_text for split_name in split_names):
            candidates.append(path)

    for path in candidates:
        if path.exists():
            examples, label_list = _parse_pos_text(_read_local_file(path))
            print(f"[dataset] loaded MasakhaPOS from {path}")
            return SimpleMasakhaPOSCorpus(examples=examples, label_list=label_list)

    raise FileNotFoundError(f"No local MasakhaPOS file found under {data_dir} for {language}/{split}.")


class MasakhaPOSDataset(Dataset):
    """MasakhaPOS token-classification dataset with word-to-subword label alignment."""

    def __init__(
        self,
        tokenizer_name_or_path: str,
        language: str,
        split: str = "train",
        max_length: int = 256,
        dataset_name: str = DEFAULT_DATASET_NAME,
        github_repo: str = DEFAULT_GITHUB_REPO,
        data_dir: str | None = None,
        ignore_index: int = IGNORE_INDEX,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)
        if not self.tokenizer.is_fast:
            raise ValueError("MasakhaPOSDataset requires a fast tokenizer so word_ids() is available.")

        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.dataset_name = dataset_name
        self.github_repo = github_repo
        self.data_dir = data_dir
        self.language = language
        self.split = split
        self.max_length = max_length
        self.ignore_index = ignore_index

        self.raw_dataset = self._load_raw_dataset()

        self.token_field = _find_existing_field(self.raw_dataset.features, TOKEN_FIELDS, "token")
        self.label_field = _find_existing_field(self.raw_dataset.features, LABEL_FIELDS, "POS label")
        self.label_info = _build_label_info(self.raw_dataset, self.label_field)
        self.label_list = self.label_info.label_list
        self.num_labels = len(self.label_list)
        self.samples = [self._encode_example(example) for example in self.raw_dataset]

    def _load_raw_dataset(self) -> Any:
        if self.data_dir is not None:
            return _load_local_dataset(Path(self.data_dir), self.language, self.split)

        if self.dataset_name in {"github", "masakhane-github"}:
            return _load_github_dataset(self.github_repo, self.language, self.split)

        try:
            return load_dataset(self.dataset_name, self.language, split=self.split)
        except Exception as exc:
            if self.dataset_name not in {"auto", DEFAULT_DATASET_NAME}:
                raise RuntimeError(
                    f"Failed to load dataset='{self.dataset_name}', language='{self.language}', split='{self.split}'."
                ) from exc

            print(
                f"[dataset] Hugging Face dataset '{self.dataset_name}' was not available for "
                f"{self.language}/{self.split}; trying official GitHub fallback."
            )
            return _load_github_dataset(self.github_repo, self.language, self.split)

    def _normalise_labels(self, labels: list[Any]) -> list[int]:
        if self.label_info.label_to_id is None:
            return [int(label) for label in labels]
        return [self.label_info.label_to_id[label] for label in labels]

    def _encode_example(self, example: dict[str, Any]) -> dict[str, Any]:
        tokens = example[self.token_field]
        labels = self._normalise_labels(example[self.label_field])

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        word_ids = encoding.word_ids()
        aligned_labels = self.align_labels(labels, word_ids)

        return {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
            "tokens": tokens,
            "word_ids": word_ids,
        }

    def align_labels(self, labels: list[int], word_ids: list[int | None]) -> list[int]:
        aligned_labels = []
        previous_word_id = None

        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(self.ignore_index)
            elif word_id != previous_word_id:
                aligned_labels.append(labels[word_id])
            else:
                aligned_labels.append(self.ignore_index)
            previous_word_id = word_id

        return aligned_labels

    def debug_example(self, index: int = 0, max_pieces: int | None = 80) -> None:
        sample = self[index]
        input_ids = sample["input_ids"].tolist()
        pieces = self.tokenizer.convert_ids_to_tokens(input_ids)
        labels = sample["labels"].tolist()
        word_ids = sample["word_ids"]

        if max_pieces is not None:
            pieces = pieces[:max_pieces]
            labels = labels[:max_pieces]
            word_ids = word_ids[:max_pieces]

        print(f"Language: {self.language} | split: {self.split} | example: {index}")
        print(f"Original words: {' '.join(sample['tokens'])}")
        print("Subword alignment:")
        for piece, word_id, label_id in zip(pieces, word_ids, labels):
            label_name = "IGN" if label_id == self.ignore_index else self.label_list[label_id]
            print(f"{piece:>16}  word_id={str(word_id):>4}  label={label_name}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]
