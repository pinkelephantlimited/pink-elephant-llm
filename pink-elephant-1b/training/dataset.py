import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional


class TextDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        tokenizer,
        max_length: int = 2048,
        stride: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.examples = []

        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            for i in range(0, len(tokens), stride):
                chunk = tokens[i : i + max_length + 1]
                if len(chunk) > 1:
                    self.examples.append(chunk)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tokens = self.examples[idx]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)

        seq_len = len(input_ids)
        if seq_len < self.max_length:
            pad_len = self.max_length - seq_len
            input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=self.tokenizer.pad_token_id)
            labels = torch.nn.functional.pad(labels, (0, pad_len), value=-100)

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


class DataCollator:
    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        input_ids = torch.stack([item["input_ids"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
