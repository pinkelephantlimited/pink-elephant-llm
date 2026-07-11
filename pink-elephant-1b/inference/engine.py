import torch
import torch.nn as nn
from typing import Optional

from model import PinkElephantForCausalLM
from config import ModelConfig
from tokenizer import PinkElephantTokenizer


class InferenceEngine:
    def __init__(
        self,
        model: PinkElephantForCausalLM,
        tokenizer: PinkElephantTokenizer,
        device: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device

        self.model = model.to(self.device).to(torch_dtype)
        self.model.eval()
        self.tokenizer = tokenizer

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
    ) -> str:
        input_ids = torch.tensor(
            [self.tokenizer.encode(prompt, add_special_tokens=True)],
            dtype=torch.long,
            device=self.device,
        )

        output_ids = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        return self.tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)

    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
    ):
        input_ids = torch.tensor(
            [self.tokenizer.encode(prompt, add_special_tokens=True)],
            dtype=torch.long,
            device=self.device,
        )

        yield prompt

        past_key_values = None

        for _ in range(max_new_tokens):
            if past_key_values is None or input_ids.shape[1] > 1:
                current_input = input_ids
                past_key_values = None
            else:
                current_input = input_ids[:, -1:]

            outputs = self.model(
                input_ids=current_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs["logits"]
            past_key_values = outputs["past_key_values"]

            next_logits = logits[:, -1, :] / temperature

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

            token_str = self.tokenizer.decode([next_token.item()], skip_special_tokens=True)
            yield token_str

            if next_token.item() == self.tokenizer.eos_token_id:
                break
