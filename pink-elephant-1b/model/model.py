import torch
import torch.nn as nn
from typing import Optional

from .layers import RMSNorm, RotaryEmbedding
from .transformer_block import TransformerBlock
from config.model_config import ModelConfig


class PinkElephantModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.02)
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.rotary_emb = RotaryEmbedding(
            config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[list[tuple[torch.Tensor, torch.Tensor]]]]:
        batch_size, seq_len = input_ids.shape

        hidden_states = self.embed_tokens(input_ids)

        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        cos, sin = self.rotary_emb(hidden_states, position_ids)

        causal_mask = self._make_causal_mask(seq_len, input_ids.device, hidden_states.dtype)
        if attention_mask is not None:
            causal_mask = causal_mask + attention_mask.unsqueeze(1).unsqueeze(2)

        new_past_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training and not use_cache:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer._forward_checkpoint,
                    hidden_states, cos, sin, causal_mask,
                    use_reentrant=False,
                )
            else:
                hidden_states, present_kv = layer(
                    hidden_states, cos, sin, causal_mask, past_kv, use_cache
                )
                if use_cache and present_kv is not None:
                    new_past_key_values.append(present_kv)

        hidden_states = self.norm(hidden_states)

        return hidden_states, new_past_key_values

    def _make_causal_mask(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1)
        return mask[None, None, :, :]


class PinkElephantForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = PinkElephantModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> dict:
        hidden_states, new_past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": new_past_key_values,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        eos_token_id: int = 50256,
    ) -> torch.Tensor:
        self.eval()
        batch_size = input_ids.shape[0]
        past_key_values = None

        for _ in range(max_new_tokens):
            if past_key_values is None or input_ids.shape[1] > 1:
                current_input = input_ids
                past_key_values = None
            else:
                current_input = input_ids[:, -1:]

            outputs = self(
                input_ids=current_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs["logits"]
            past_key_values = outputs["past_key_values"]

            next_logits = logits[:, -1, :]

            next_logits = next_logits / temperature

            for i in range(batch_size):
                for token_id in set(input_ids[i].tolist()):
                    if next_logits[i, token_id] < 0:
                        next_logits[i, token_id] *= repetition_penalty
                    else:
                        next_logits[i, token_id] /= repetition_penalty

            if top_k > 0:
                values, _ = torch.topk(next_logits, top_k, dim=-1)
                next_logits[next_logits < values[:, -1:]] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_logits[indices_to_remove] = float("-inf")

            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if (next_token == eos_token_id).any():
                break

        return input_ids
