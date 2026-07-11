import json
import os


def create_base_vocab(save_path: str | None = None) -> dict[str, int]:
    vocab = {}
    for i in range(256):
        vocab[bytes([i]).decode("latin-1")] = i

    special_tokens = {
        "<|endoftext|>": 50256,
        "<|pad|>": 50257,
        "<|bos|>": 50258,
        "<|eos|>": 50259,
        "<|unk|>": 50260,
    }

    for i in range(256, 50256):
        vocab[f"<|token_{i}|>"] = i

    for token, idx in special_tokens.items():
        vocab[token] = idx

    if save_path:
        dirname = os.path.dirname(save_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)

    return vocab
