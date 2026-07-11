import json
import re
from pathlib import Path
from typing import Optional


class PinkElephantTokenizer:
    def __init__(
        self,
        vocab_file: str | None = None,
        vocab: dict[str, int] | None = None,
        merges_file: str | None = None,
    ):
        if vocab_file:
            with open(vocab_file) as f:
                self.vocab = json.load(f)
        elif vocab:
            self.vocab = vocab
        else:
            self.vocab = {}
            for i in range(256):
                self.vocab[bytes([i]).decode("latin-1")] = i
            for i in range(256, 50256):
                self.vocab[f"<|token_{i}|>"] = i
            self.vocab["<|endoftext|>"] = 50256
            self.vocab["<|pad|>"] = 50257
            self.vocab["<|bos|>"] = 50258
            self.vocab["<|eos|>"] = 50259
            self.vocab["<|unk|>"] = 50260

        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        self.bos_token = "<|bos|>"
        self.eos_token = "<|eos|>"
        self.pad_token = "<|pad|>"
        self.unk_token = "<|unk|>"
        self.bos_token_id = self.vocab.get(self.bos_token, 50258)
        self.eos_token_id = self.vocab.get(self.eos_token, 50259)
        self.pad_token_id = self.vocab.get(self.pad_token, 50257)
        self.unk_token_id = self.vocab.get(self.unk_token, 50260)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        tokens = []
        if add_special_tokens:
            tokens.append(self.bos_token_id)

        byte_repr = text.encode("utf-8").decode("latin-1")
        for char in byte_repr:
            if char in self.vocab:
                tokens.append(self.vocab[char])
            else:
                tokens.append(self.unk_token_id)

        if add_special_tokens:
            tokens.append(self.eos_token_id)

        return tokens

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        chars = []
        for tid in token_ids:
            if skip_special_tokens and tid in (
                self.bos_token_id,
                self.eos_token_id,
                self.pad_token_id,
            ):
                continue
            if tid < 256:
                chars.append(self.ids_to_tokens.get(tid, chr(tid)))
            else:
                token = self.ids_to_tokens.get(tid, self.unk_token)
                if token.startswith("<|") and token.endswith("|>"):
                    continue
                chars.append(token)

        byte_str = "".join(chars)
        try:
            return byte_str.encode("latin-1", errors="replace").decode("utf-8", errors="replace")
        except Exception:
            return byte_str

    def __len__(self) -> int:
        return self.vocab_size

    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_pretrained(cls, path: str) -> "PinkElephantTokenizer":
        return cls(vocab_file=path)
