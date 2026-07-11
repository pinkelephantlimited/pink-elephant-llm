import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tokenizer import PinkElephantTokenizer, create_base_vocab


def test_vocab_creation():
    vocab = create_base_vocab()
    assert len(vocab) >= 50257
    assert "<|endoftext|>" in vocab


def test_tokenizer_encode_decode():
    tokenizer = PinkElephantTokenizer()
    text = "Hello, world!"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    assert len(ids) > 0
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)


def test_special_tokens():
    tokenizer = PinkElephantTokenizer()
    text = "test"
    ids_with_special = tokenizer.encode(text, add_special_tokens=True)
    assert ids_with_special[0] == tokenizer.bos_token_id
    assert ids_with_special[-1] == tokenizer.eos_token_id

    ids_without_special = tokenizer.encode(text, add_special_tokens=False)
    assert ids_without_special[0] != tokenizer.bos_token_id


def test_vocab_size():
    tokenizer = PinkElephantTokenizer()
    assert len(tokenizer) >= 50257


def test_save_load():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write('{"hello": 0, "world": 1, "<|bos|>": 2, "<|eos|>": 3, "<|pad|>": 4, "<|unk|>": 5}')
        temp_path = f.name

    tokenizer = PinkElephantTokenizer(vocab_file=temp_path)
    assert tokenizer.vocab_size == 6

    save_path = tempfile.mktemp(suffix=".json")
    tokenizer.save(save_path)
    assert os.path.exists(save_path)

    loaded = PinkElephantTokenizer(vocab_file=save_path)
    assert loaded.vocab_size == 6

    os.unlink(temp_path)
    os.unlink(save_path)


if __name__ == "__main__":
    test_vocab_creation()
    print("✓ test_vocab_creation")
    test_tokenizer_encode_decode()
    print("✓ test_tokenizer_encode_decode")
    test_special_tokens()
    print("✓ test_special_tokens")
    test_vocab_size()
    print("✓ test_vocab_size")
    test_save_load()
    print("✓ test_save_load")
    print("\nAll tokenizer tests passed!")
