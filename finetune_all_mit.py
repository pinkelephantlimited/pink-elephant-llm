# /// script
# dependencies = ["transformers", "datasets", "accelerate", "huggingface_hub", "peft", "torch", "bitsandbytes"]
# ///

# ---
# jupyter:
#   marimo:
#     name: "Fine-tune ALL MIT Models on molab"
# ---

# %% [markdown]
# # Fine-tune ALL MIT-Licensed Models — Max Quality
#
# | Model | Method | Why |
# |-------|--------|-----|
# | GPT-2 Small (124M) → Phi-2 (2.7B) | **Full fine-tune** | Updates all weights — highest quality |
# | Phi-3 Mini (3.8B) → Phi-4 (14B) | **LoRA rank=128** | Too large for full fine-tune on single GPU |
#
# **Data**: FineWeb-Edu (300K) + OpenWebMath (100K) + CodeParrot (100K) — only the 3 highest-quality sources.

# %% [markdown]
# ## 1. Install

# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "torch", "transformers", "datasets", "accelerate",
    "huggingface_hub", "peft", "bitsandbytes", "sentencepiece"])
print("Installed!")

# %% [markdown]
# ## 2. Login

# %%
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json, shutil, gc
import torch
from huggingface_hub import login, HfApi
import getpass

token = getpass.getpass("HF token: ")
login(token, add_to_git_credential=False)
api = HfApi()
print("Logged in!")

# %% [markdown]
# ## 3. Config

# %%
# (base_hf_id, target_repo_slug, display_name, params, method)
# method: "full" = full fine-tune, "lora" = LoRA rank=128
MODELS = [
    ("gpt2",                "pink-elephant-gpt2-small",   "GPT-2 Small",  "124M",  "full"),
    ("gpt2-medium",         "pink-elephant-gpt2-medium",  "GPT-2 Medium", "355M",  "full"),
    ("gpt2-large",          "pink-elephant-gpt2-large",   "GPT-2 Large",  "774M",  "full"),
    ("gpt2-xl",             "pink-elephant-gpt2-xl",      "GPT-2 XL",     "1.5B",  "full"),
    ("microsoft/phi-1_5",   "pink-elephant-phi-1_5",      "Phi-1.5",      "1.3B",  "full"),
    ("microsoft/phi-2",     "pink-elephant-phi-2",        "Phi-2",        "2.7B",  "full"),
    ("microsoft/Phi-3-mini-4k-instruct", "pink-elephant-phi-3-mini", "Phi-3 Mini", "3.8B", "lora"),
    ("microsoft/Phi-4",     "pink-elephant-phi-4",        "Phi-4",        "14B",   "lora"),
]

# %% [markdown]
# ## 4. Load Data — Highest quality sources only

# %%
from datasets import load_dataset
import random

train_texts = []

def add_from_ds(ds, key, limit):
    count = 0
    for x in ds:
        if count >= limit:
            break
        if key in x and x[key] and isinstance(x[key], str):
            train_texts.append(x[key][:2048])
            count += 1
    return count

print("Loading high-quality data...")

print("  FineWeb-Edu (educational web text)...")
ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
add_from_ds(ds, "text", 300000)

print("  OpenWebMath (mathematical reasoning)...")
ds = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
add_from_ds(ds, "text", 100000)

print("  CodeParrot (multi-language code)...")
ds = load_dataset("transformersbook/codeparrot", split="train", streaming=True)
count = 0
for x in ds:
    if count >= 100000:
        break
    if "content" in x and x["content"] and isinstance(x["content"], str):
        train_texts.append(x["content"][:2048])
        count += 1

random.seed(42)
random.shuffle(train_texts)
print(f"Total: {len(train_texts):,} examples")

# %% [markdown]
# ## 5. Fine-Tune

# %%
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

def build_readme(base_id, target_repo, display, params, method):
    method_note = "Full parameter fine-tune (all weights updated)" if method == "full" else "LoRA rank=128 (adapter fine-tune)"
    return f"""---
tags:
- pink-elephant
- finetuned
- mit-license
license: mit
language:
- en
pipeline_tag: text-generation
library_name: transformers
---

# 🐘 Pink Elephant {display}

A fine-tuned version of **{base_id}** ({params} parameters). {method_note}.

**Base model**: [{base_id}](https://huggingface.co/{base_id})
**Fine-tuning script**: [finetune_all_mit.py](finetune_all_mit.py)

## Training Data

| Source | Type | Examples |
|--------|------|----------|
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | Educational web text | 300,000 |
| [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math) | Mathematical reasoning | 100,000 |
| [CodeParrot](https://huggingface.co/datasets/transformersbook/codeparrot) | Code (multi-language) | 100,000 |
| **Total** | | **500,000** |

## Usage

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="pinkelephantlimited/{target_repo}")
output = pipe("The definition of machine learning is", max_new_tokens=80)[0]["generated_text"]
print(output)
```

## License

MIT
"""

def fine_tune_one(base_id, target_repo, display, params, method):
    print(f"\n{'='*60}")
    print(f"{'='*60}")
    print(f"Fine-tuning {display} ({params}) — {method.upper()}")
    print(f"Base: {base_id}")
    print(f"Target: pinkelephantlimited/{target_repo}")
    print(f"{'='*60}\n")
    gc.collect()
    torch.cuda.empty_cache()

    # Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize
    def encode(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding=False)

    dataset = Dataset.from_dict({"text": train_texts})
    dataset = dataset.map(encode, remove_columns=["text"], desc="Tokenizing")
    dataset = dataset.filter(lambda x: len(x["input_ids"]) > 10)
    print(f"Tokenized: {len(dataset)} examples")

    # Model
    print(f"Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )

    # LoRA setup for large models
    if method == "lora":
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        if "gpt2" in base_id.lower():
            target_modules = ["c_attn", "c_proj", "c_fc"]
        lora_config = LoraConfig(
            r=128, lora_alpha=256, target_modules=target_modules,
            lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    # Train
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    args = TrainingArguments(
        output_dir=f"/tmp/ft_{target_repo}",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        bf16=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=collator)
    trainer.train()

    # Save
    save_dir = f"/tmp/{target_repo}"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    print("Saving model...")
    if method == "lora":
        merged = model.merge_and_unload()
        merged.save_pretrained(save_dir, safe_serialization=True)
        del merged
    else:
        model.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)

    # Fix config
    cfg_path = os.path.join(save_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["_name_or_path"] = f"pinkelephantlimited/{target_repo}"
    cfg.pop("_attn_implementation_autoset", None)
    if "phi-4" in base_id.lower():
        cfg["architectures"] = ["Phi4ForCausalLM"]
    elif "phi-3" in base_id.lower():
        cfg["architectures"] = ["Phi3ForCausalLM"]
    elif "phi-2" in base_id.lower() or "phi-1" in base_id.lower():
        cfg["architectures"] = ["PhiForCausalLM"]
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    # Upload
    repo_id = f"pinkelephantlimited/{target_repo}"
    print(f"Uploading to {repo_id}...")
    api.create_repo(repo_id, private=False, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=save_dir, repo_id=repo_id, ignore_patterns=["*.bin"])
    readme = build_readme(base_id, target_repo, display, params, method)
    api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md", repo_id=repo_id, repo_type="model")
    print(f"✓ {display} done → https://huggingface.co/{repo_id}")

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

# %% [markdown]
# ## 6. Run

# %%
for base_id, target, display, params, method in MODELS:
    fine_tune_one(base_id, target, display, params, method)

print("\n✓ ALL MODELS FINISHED!")
for _, target, display, params, _ in MODELS:
    print(f"  https://huggingface.co/pinkelephantlimited/{target} — {display} {params}")
