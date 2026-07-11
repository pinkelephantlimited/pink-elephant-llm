import torch
import torch.nn as nn

from .layers import RMSNorm, SwiGLU
from .attention import GroupedQueryAttention


class TransformerBlock(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_hidden_layers
        self.self_attn = GroupedQueryAttention(config, layer_idx)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size, layer_idx, config.num_hidden_layers)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        residual = x
        x = self.input_layernorm(x)
        x, present_kv = self.self_attn(x, cos, sin, attention_mask, past_key_value, use_cache)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x, present_kv

    def _forward_checkpoint(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x, _ = self.forward(x, cos, sin, attention_mask, past_key_value=None, use_cache=False)
        return x
