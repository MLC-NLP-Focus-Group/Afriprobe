import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer


IGNORE_INDEX = -100


class MasakhaPOSDataset(Dataset):
    def __init__(
        self,
        tokenizer_name: str,
        language: str,
        split: str = "train",
        max_length: int = 256,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.language = language
        self.split = split
        self.max_length = max_length

        # load dataset
        self.data = load_dataset("masakhapos", language, split=split)

        self.samples = []
        self._build()

    def _build(self):
        for item in self.data:
            tokens = item["tokens"]
            labels = item["labels"]

            enc = self.tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors=None,
            )

            word_ids = enc.word_ids()

            aligned_labels = self.align_labels(labels, word_ids)

            self.samples.append({
                "input_ids": torch.tensor(enc["input_ids"]),
                "attention_mask": torch.tensor(enc["attention_mask"]),
                "labels": torch.tensor(aligned_labels),
                "word_ids": word_ids,
                "tokens": tokens,
            })

    def align_labels(self, labels, word_ids):
        """
        Map word-level labels → subword-level labels
        Only first subword keeps label, rest = -100
        """
        aligned = []
        prev_word_id = None

        for word_id in word_ids:
            if word_id is None:
                aligned.append(IGNORE_INDEX)
            elif word_id != prev_word_id:
                aligned.append(labels[word_id])
            else:
                aligned.append(IGNORE_INDEX)

            prev_word_id = word_id

        return aligned

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]