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
# Load once, fine-tune every MIT model (124M → 14B), upload each to its own HF repo with professional README.
# Uses LoRA for efficiency — even 14B fits on Blackwell 96GB.
#
# **Models**:
# - GPT-2 Small (124M) → Medium (355M) → Large (774M) → XL (1.5B)
# - Phi-1.5 (1.3B) → Phi-2 (2.7B) → Phi-3 Mini (3.8B) → Phi-4 (14B)
#
# **Quality settings**: 100K examples, 3 epochs, rank=64 LoRA, cosine LR schedule, all 7 datasets.

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
# ## 3. Config — Models to fine-tune

# %%
# (base_hf_id, target_repo_slug, display_name, params)
MODELS = [
    ("gpt2",                "pink-elephant-gpt2-small",   "GPT-2 Small",  "124M"),
    ("gpt2-medium",         "pink-elephant-gpt2-medium",  "GPT-2 Medium", "355M"),
    ("gpt2-large",          "pink-elephant-gpt2-large",   "GPT-2 Large",  "774M"),
    ("gpt2-xl",             "pink-elephant-gpt2-xl",      "GPT-2 XL",     "1.5B"),
    ("microsoft/phi-1_5",   "pink-elephant-phi-1_5",      "Phi-1.5",      "1.3B"),
    ("microsoft/phi-2",     "pink-elephant-phi-2",        "Phi-2",        "2.7B"),
    ("microsoft/Phi-3-mini-4k-instruct", "pink-elephant-phi-3-mini", "Phi-3 Mini", "3.8B"),
    ("microsoft/Phi-4",     "pink-elephant-phi-4",        "Phi-4",        "14B"),
]

# %% [markdown]
# ## 4. Load Dataset — All 7 sources, max quality

# %%
from datasets import load_dataset

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

print("Loading all 7 datasets...")

print("  FineWeb-Edu...")
ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
add_from_ds(ds, "text", 25000)

print("  FineWeb...")
ds = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True)
add_from_ds(ds, "text", 25000)

print("  OpenWebMath...")
ds = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
add_from_ds(ds, "text", 15000)

print("  SmolLM Corpus...")
ds = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)
count = 0
for x in ds:
    if count >= 15000:
        break
    for f in ["text", "content"]:
        if f in x and x[f] and isinstance(x[f], str):
            train_texts.append(x[f][:2048])
            count += 1
            break

print("  CodeParrot...")
ds = load_dataset("transformersbook/codeparrot", split="train", streaming=True)
count = 0
for x in ds:
    if count >= 10000:
        break
    if "content" in x and x["content"] and isinstance(x["content"], str):
        train_texts.append(x["content"][:2048])
        count += 1

print("  Nemotron-Legal...")
ds = load_dataset("nvidia/Nemotron-Pretraining-Legal-v1",
                  "Nemotron-Pretraining-Legal-Case-Law-Summary",
                  split="train", streaming=True)
count = 0
for x in ds:
    if count >= 5000:
        break
    for f in ["text", "input", "content"]:
        if f in x and x[f] and isinstance(x[f], str):
            train_texts.append(x[f][:2048])
            count += 1
            break

print("  Investopedia...")
ds = load_dataset("infCapital/investopedia_terms_en", split="train", streaming=True)
add_from_ds(ds, "text", 5000)

import random
random.seed(42)
random.shuffle(train_texts)
train_texts = train_texts[:100000]
print(f"Total: {len(train_texts)} examples")

# %% [markdown]
# ## 5. Fine-Tune Loop — High Quality

# %%
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

def build_readme(base_id, target_repo, display, params):
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

A fine-tuned version of **{base_id}** ({params} parameters), trained on a diverse corpus of 7 datasets covering general web text, mathematics, code, legal, and finance domains.

**Base model**: [{base_id}](https://huggingface.co/{base_id})
**Fine-tuning script**: [finetune_all_mit.py](finetune_all_mit.py)

## Model Details

| Property | Value |
|----------|-------|
| Base model | {base_id} |
| Parameters | {params} |
| License | MIT |
| Fine-tuning data | FineWeb-Edu, FineWeb, OpenWebMath, SmolLM, CodeParrot, Nemotron-Legal, Investopedia |
| Training examples | 100,000 |
| Epochs | 3 |
| LoRA rank | 64 |
| Sequence length | 512 |

## Usage

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="pinkelephantlimited/{target_repo}")
output = pipe("The definition of machine learning is", max_new_tokens=80)[0]["generated_text"]
print(output)
```

## Training Philosophy

Fine-tuned from a permissively licensed MIT base model. The resulting weights are fully open and owned by Pink Elephant Limited.

## License

MIT
"""

def fine_tune_one(base_id, target_repo, display, params):
    print(f"\n{'='*60}")
    print(f"Fine-tuning {display} ({params}) — {base_id}")
    print(f"Target: pinkelephantlimited/{target_repo}")
    print(f"{'='*60}\n")
    gc.collect()
    torch.cuda.empty_cache()

    # Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize — full 100K
    def encode(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding=False)

    dataset = Dataset.from_dict({"text": train_texts})
    dataset = dataset.map(encode, remove_columns=["text"], desc="Tokenizing", num_proc=2)
    dataset = dataset.filter(lambda x: len(x["input_ids"]) > 10)

    # Model
    print(f"Loading model ({params})...")
    model = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )

    # LoRA
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if "gpt2" in base_id.lower():
        target_modules = ["c_attn", "c_proj", "c_fc"]

    lora_config = LoraConfig(
        r=64, lora_alpha=128, target_modules=target_modules,
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

    # Merge + save
    save_dir = f"/tmp/{target_repo}"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    print("Saving merged model...")
    merged = model.merge_and_unload()
    merged.save_pretrained(save_dir, safe_serialization=True)
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
    elif "gpt2" in base_id.lower():
        cfg["architectures"] = ["GPT2LMHeadModel"]
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    # Upload model
    repo_id = f"pinkelephantlimited/{target_repo}"
    print(f"Uploading model to {repo_id}...")
    api.create_repo(repo_id, private=False, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=save_dir, repo_id=repo_id, ignore_patterns=["*.bin"])

    # Upload README
    readme = build_readme(base_id, target_repo, display, params)
    api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md", repo_id=repo_id, repo_type="model")

    print(f"✓ {display} done → https://huggingface.co/{repo_id}")

    del model, merged, trainer
    gc.collect()
    torch.cuda.empty_cache()

# %% [markdown]
# ## 6. Run

# %%
for base_id, target, display, params in MODELS:
    fine_tune_one(base_id, target, display, params)

print("\n✓ ALL 8 MODELS FINISHED!")
print("Repos:")
for _, target, display, params in MODELS:
    print(f"  https://huggingface.co/pinkelephantlimited/{target} — {display} {params}")
