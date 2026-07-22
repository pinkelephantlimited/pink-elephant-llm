# /// script
# dependencies = ["transformers", "torch", "huggingface_hub"]
# ///

# ---
# jupyter:
#   marimo:
#     name: "Inference - Pink Elephant 22M"
# ---

# %% [markdown]
# # 🐘 Pink Elephant 22M — Interactive Inference
#
# Load the model and type your own prompts.

# %%
import marimo as mo
import torch
from transformers import pipeline
import os

# %% [markdown]
# ## Load Model

# %%
MODEL = "pinkelephantlimited/pink-elephant-22m"
pipe = pipeline("text-generation", model=MODEL, trust_remote_code=True)
mo.output.replace(mo.md(f"**✅ Model loaded: {MODEL}**"))

# %% [markdown]
# ## Enter Your Prompt

# %%
prompt = mo.ui.text(placeholder="def fibonacci(n):", label="Your prompt")
max_tokens = mo.ui.slider(start=10, stop=200, step=10, value=60, label="Max new tokens")
temperature = mo.ui.slider(start=0.1, stop=1.5, step=0.1, value=0.5, label="Temperature")

mo.hstack([prompt, max_tokens, temperature], justify="start")

# %% [markdown]
# ## Generate

# %%
generate_button = mo.ui.run_button(label="Generate")

# %%
mo.ui.run(
    generate_button,
    lambda: mo.output.replace(
        mo.md(
            f"**Prompt:** `{prompt.value}`\n\n"
            f"**Output:**\n```\n{pipe(prompt.value, max_new_tokens=max_tokens.value, do_sample=True, temperature=temperature.value)[0]['generated_text']}\n```"
        )
    )
)
