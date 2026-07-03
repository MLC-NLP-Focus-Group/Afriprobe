from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from datasets import ClassLabel, Sequence, load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


IGNORE_INDEX = -100
DEFAULT_DATASET_NAME = "masakhapos"
TOKEN_FIELDS = ("tokens", "words", "word")
LABEL_FIELDS = ("labels", "pos_tags", "upos", "upos_tags", "tags")


@dataclass(frozen=True)
class LabelInfo:
    field_name: str
    label_list: list[str]
    label_to_id: dict[str, int] | None


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


class MasakhaPOSDataset(Dataset):
    """MasakhaPOS token-classification dataset with word-to-subword label alignment."""

    def __init__(
        self,
        tokenizer_name_or_path: str,
        language: str,
        split: str = "train",
        max_length: int = 256,
        dataset_name: str = DEFAULT_DATASET_NAME,
        ignore_index: int = IGNORE_INDEX,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)
        if not self.tokenizer.is_fast:
            raise ValueError("MasakhaPOSDataset requires a fast tokenizer so word_ids() is available.")

        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.dataset_name = dataset_name
        self.language = language
        self.split = split
        self.max_length = max_length
        self.ignore_index = ignore_index

        try:
            self.raw_dataset = load_dataset(dataset_name, language, split=split)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load dataset='{dataset_name}', language='{language}', split='{split}'."
            ) from exc

        self.token_field = _find_existing_field(self.raw_dataset.features, TOKEN_FIELDS, "token")
        self.label_field = _find_existing_field(self.raw_dataset.features, LABEL_FIELDS, "POS label")
        self.label_info = _build_label_info(self.raw_dataset, self.label_field)
        self.label_list = self.label_info.label_list
        self.num_labels = len(self.label_list)
        self.samples = [self._encode_example(example) for example in self.raw_dataset]

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
