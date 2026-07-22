# /// script
# dependencies = ["transformers", "datasets", "accelerate", "huggingface_hub", "peft", "torch", "bitsandbytes", "sentencepiece"]
# ///

# ---
# jupyter:
#   marimo:
#     name: "Fine-tune ALL MIT Models on T4 (Colab)"
# ---

# %% [markdown]
# # Fine-tune ALL MIT Models — T4 (16GB) Optimized
#
# | Model | Method | Why |
# |-------|--------|-----|
# | GPT-2 Small / Medium / Large | **Full fine-tune** | Fits in fp16/batch=1 |
# | GPT-2 XL / Phi-1.5 | **LoRA rank=64** (fp16) | Don't fit full |
# | Phi-2 / Phi-3 Mini / Phi-4 | **QLoRA rank=64** (4-bit) | Too large for fp16 |
#
# T4 has no bf16 support — uses fp16.

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
# method: "full" | "lora" (fp16 LoRA) | "qlora" (4-bit QLoRA)
MODELS = [
    ("gpt2",                "pink-elephant-gpt2-small-t4",   "GPT-2 Small",  "124M",  "full"),
    ("gpt2-medium",         "pink-elephant-gpt2-medium-t4",  "GPT-2 Medium", "355M",  "full"),
    ("gpt2-large",          "pink-elephant-gpt2-large-t4",   "GPT-2 Large",  "774M",  "full"),
    ("gpt2-xl",             "pink-elephant-gpt2-xl-t4",      "GPT-2 XL",     "1.5B",  "lora"),
    ("microsoft/phi-1_5",   "pink-elephant-phi-1_5-t4",      "Phi-1.5",      "1.3B",  "lora"),
    ("microsoft/phi-2",     "pink-elephant-phi-2-t4",        "Phi-2",        "2.7B",  "qlora"),
    ("microsoft/Phi-3-mini-4k-instruct", "pink-elephant-phi-3-mini-t4", "Phi-3 Mini", "3.8B", "qlora"),
    ("microsoft/Phi-4",     "pink-elephant-phi-4-t4",        "Phi-4",        "14B",   "qlora"),
]

# %% [markdown]
# ## 4. Load Data (same as max quality — 500K from top 3)

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
print("  FineWeb-Edu...")
ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
add_from_ds(ds, "text", 300000)

print("  OpenWebMath...")
ds = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
add_from_ds(ds, "text", 100000)

print("  CodeParrot...")
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
    DataCollatorForLanguageModeling, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset

def build_readme(base_id, target_repo, display, params, method):
    notes = {
        "full": "Full parameter fine-tune (fp16)",
        "lora": "LoRA rank=64 (fp16, adapter)",
        "qlora": "QLoRA rank=64 (4-bit NF4 quantized, adapter)",
    }
    return f"""---
tags:
- pink-elephant
- finetuned
- mit-license
- t4
license: mit
language:
- en
pipeline_tag: text-generation
library_name: transformers
---

# 🐘 Pink Elephant {display} (T4)

A fine-tuned version of **{base_id}** ({params} parameters), optimized for T4 GPUs. {notes[method]}.

**Base model**: [{base_id}](https://huggingface.co/{base_id})
**Fine-tuning script**: [finetune_all_mit_t4.py](finetune_all_mit_t4.py)

## Training Data

| Source | Type | Examples |
|--------|------|----------|
| FineWeb-Edu | Educational web text | 300,000 |
| OpenWebMath | Mathematical reasoning | 100,000 |
| CodeParrot | Code (multi-language) | 100,000 |
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

def get_lora_modules(base_id):
    n = base_id.lower()
    if "gpt2" in n:
        return ["c_attn", "c_proj", "c_fc"]
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

def fine_tune_one(base_id, target_repo, display, params, method):
    print(f"\n{'='*60}")
    print(f"Fine-tuning {display} ({params}) — {method.upper()}")
    print(f"Base: {base_id}  →  pinkelephantlimited/{target_repo}")
    print(f"{'='*60}\n")
    gc.collect()
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def encode(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding=False)

    dataset = Dataset.from_dict({"text": train_texts})
    dataset = dataset.map(encode, remove_columns=["text"], desc="Tokenizing")
    dataset = dataset.filter(lambda x: len(x["input_ids"]) > 10)
    print(f"Tokenized: {len(dataset)} examples")

    # Load model
    kw = dict(torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    if method == "qlora":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(base_id, **kw)

    # Setup LoRA/QLoRA
    if method in ("lora", "qlora"):
        if method == "qlora":
            model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=64, lora_alpha=128, target_modules=get_lora_modules(base_id),
            lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    # Batch sizes tuned for T4 16GB
    batch_map = {
        "full": {"bs": 1, "ga": 8},
        "lora": {"bs": 1, "ga": 4},
        "qlora": {"bs": 1, "ga": 2},
    }
    bm = batch_map[method]
    # GPT-2 Small can use bigger batch
    if method == "full" and params == "124M":
        bm = {"bs": 4, "ga": 4}

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    args = TrainingArguments(
        output_dir=f"/tmp/ft_{target_repo}",
        per_device_train_batch_size=bm["bs"],
        gradient_accumulation_steps=bm["ga"],
        num_train_epochs=3,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        fp16=True,
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
    if method in ("lora", "qlora"):
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

print("\n✓ ALL MODELS FINISHED ON T4!")
for _, target, display, params, _ in MODELS:
    print(f"  https://huggingface.co/pinkelephantlimited/{target} — {display} {params}")
