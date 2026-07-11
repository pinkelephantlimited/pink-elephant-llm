from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    vocab_size: int = 50264
    hidden_size: int = 2048
    num_hidden_layers: int = 20
    num_attention_heads: int = 16
    num_key_value_heads: Optional[int] = None
    intermediate_size: int = 8192
    hidden_act: str = "silu"
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None
    use_cache: bool = True
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02
    bos_token_id: int = 50256
    eos_token_id: int = 50256
    pad_token_id: int = 50256
    torch_dtype: str = "float32"

    @property
    def num_params_estimate(self) -> int:
        vocab = self.vocab_size * self.hidden_size
        per_layer = (
            4 * self.hidden_size * self.hidden_size
            + 2 * self.hidden_size * self.intermediate_size
            + 2 * self.hidden_size
        )
        layers = self.num_hidden_layers * per_layer
        head = self.vocab_size * self.hidden_size if not self.tie_word_embeddings else 0
        return vocab + layers + head

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


def PinkElephant1BConfig() -> ModelConfig:
    return ModelConfig(
        vocab_size=50264,
        hidden_size=2048,
        num_hidden_layers=20,
        num_attention_heads=16,
        num_key_value_heads=16,
        intermediate_size=8192,
        hidden_act="silu",
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        use_cache=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        tie_word_embeddings=True,
        initializer_range=0.02,
        bos_token_id=50258,
        eos_token_id=50259,
        pad_token_id=50257,
        torch_dtype="bfloat16",
    )


def PinkElephant50MConfig() -> ModelConfig:
    return ModelConfig(
        vocab_size=50264,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        num_key_value_heads=4,
        intermediate_size=2048,
        hidden_act="silu",
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        use_cache=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        tie_word_embeddings=True,
        initializer_range=0.02,
        bos_token_id=50258,
        eos_token_id=50259,
        pad_token_id=50257,
    )


def PinkElephant400MConfig() -> ModelConfig:
    return ModelConfig(
        vocab_size=50264,
        hidden_size=1536,
        num_hidden_layers=10,
        num_attention_heads=12,
        num_key_value_heads=4,
        intermediate_size=6144,
        hidden_act="silu",
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        use_cache=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        tie_word_embeddings=True,
        initializer_range=0.02,
        bos_token_id=50258,
        eos_token_id=50259,
        pad_token_id=50257,
        torch_dtype="bfloat16",
    )
