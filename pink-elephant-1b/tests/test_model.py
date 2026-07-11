import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.model_config import ModelConfig, PinkElephant1BConfig
from model import PinkElephantForCausalLM, PinkElephantModel
from model.layers import RMSNorm, SwiGLU, RotaryEmbedding, apply_rotary_pos_emb
from model.attention import GroupedQueryAttention


def test_rmsnorm():
    config = PinkElephant1BConfig()
    norm = RMSNorm(config.hidden_size)
    x = torch.randn(2, 4, config.hidden_size)
    out = norm(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()


def test_swiglu():
    config = PinkElephant1BConfig()
    swiglu = SwiGLU(config.hidden_size, config.intermediate_size)
    x = torch.randn(2, 4, config.hidden_size)
    out = swiglu(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()


def test_rotary_embedding():
    config = PinkElephant1BConfig()
    head_dim = config.hidden_size // config.num_attention_heads
    rotary = RotaryEmbedding(head_dim)
    x = torch.randn(2, 4, config.hidden_size)
    pos_ids = torch.arange(4).unsqueeze(0).expand(2, -1)
    cos, sin = rotary(x, pos_ids)
    assert cos.shape == (2, 4, head_dim)
    assert sin.shape == (2, 4, head_dim)


def test_attention():
    config = PinkElephant1BConfig()
    attn = GroupedQueryAttention(config, layer_idx=0)
    batch, seq, dim = 2, 8, config.hidden_size
    x = torch.randn(batch, seq, dim)
    head_dim = config.hidden_size // config.num_attention_heads
    cos = torch.randn(batch, seq, head_dim)
    sin = torch.randn(batch, seq, head_dim)
    out, kv = attn(x, cos, sin)
    assert out.shape == (batch, seq, dim)


def test_model_forward():
    config = PinkElephant1BConfig()
    config.num_hidden_layers = 2
    model = PinkElephantForCausalLM(config)
    batch, seq = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch, seq))
    outputs = model(input_ids=input_ids, labels=input_ids)
    assert "loss" in outputs
    assert outputs["loss"] is not None
    assert outputs["logits"].shape == (batch, seq, config.vocab_size)


def test_model_generate():
    config = PinkElephant1BConfig()
    config.num_hidden_layers = 2
    model = PinkElephantForCausalLM(config)
    batch, seq = 1, 4
    input_ids = torch.randint(0, config.vocab_size, (batch, seq))
    output = model.generate(input_ids, max_new_tokens=5, temperature=1.0)
    assert output.shape[1] > seq
    assert output.shape[0] == batch


def test_tied_embeddings():
    config = PinkElephant1BConfig()
    config.tie_word_embeddings = True
    model = PinkElephantForCausalLM(config)
    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_pink_elephant_config():
    config = PinkElephant1BConfig()
    assert config.hidden_size == 2048
    assert config.num_hidden_layers == 20
    assert config.num_attention_heads == 16
    assert config.intermediate_size == 8192
    params = config.num_params_estimate
    assert abs(params - 1_000_000_000) < 500_000_000


if __name__ == "__main__":
    test_rmsnorm()
    print("✓ test_rmsnorm")
    test_swiglu()
    print("✓ test_swiglu")
    test_rotary_embedding()
    print("✓ test_rotary_embedding")
    test_attention()
    print("✓ test_attention")
    test_model_forward()
    print("✓ test_model_forward")
    test_model_generate()
    print("✓ test_model_generate")
    test_tied_embeddings()
    print("✓ test_tied_embeddings")
    test_pink_elephant_config()
    print("✓ test_pink_elephant_config")
    print("\nAll tests passed!")
