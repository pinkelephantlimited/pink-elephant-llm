# /// script
# dependencies = ["transformers", "datasets", "accelerate", "huggingface_hub", "peft", "torch", "bitsandbytes"]
# ///

# ---
# jupyter:
#   marimo:
#     name: "Fine-tune ALL MIT Models on molab"
# ---

# %% [markdown]
# # Fine-tune ALL MIT-Licensed Models
#
# Load once, fine-tune every MIT model (124M → 14B), upload each to its own HF repo.
# Uses LoRA for efficiency — even 14B fits on Blackwell 96GB.
#
# **Models**:
# - GPT-2 Small (124M) → Medium (355M) → Large (774M) → XL (1.5B)
# - Phi-1.5 (1.3B) → Phi-2 (2.7B) → Phi-3 Mini (3.8B) → Phi-4 (14B)

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
import json, shutil, time, gc
import torch
from huggingface_hub import login, HfApi
import getpass

token = getpass.getpass("HF token: ")
login(token, add_to_git_credential=False)
api = HfApi()
print("Logged in!")

# %% [markdown]
# ## 3. Config — Add/remove models here

# %%
# Each entry: (hf_model_id, target_repo_name, description)
MODELS = [
    # GPT-2 family (all MIT)
    ("gpt2",                "pink-elephant-finetuned-gpt2-small",  "GPT-2 Small 124M"),
    ("gpt2-medium",         "pink-elephant-finetuned-gpt2-medium", "GPT-2 Medium 355M"),
    ("gpt2-large",          "pink-elephant-finetuned-gpt2-large",  "GPT-2 Large 774M"),
    ("gpt2-xl",             "pink-elephant-finetuned-gpt2-xl",     "GPT-2 XL 1.5B"),

    # Phi family (all MIT)
    ("microsoft/phi-1_5",   "pink-elephant-finetuned-phi-1_5",     "Phi-1.5 1.3B"),
    ("microsoft/phi-2",     "pink-elephant-finetuned-phi-2",       "Phi-2 2.7B"),
    ("microsoft/Phi-3-mini-4k-instruct", "pink-elephant-finetuned-phi-3-mini", "Phi-3 Mini 3.8B"),
    ("microsoft/Phi-4",     "pink-elephant-finetuned-phi-4",       "Phi-4 14B"),
]

# %% [markdown]
# ## 4. Load Dataset (once, shared across all models)

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

MAX_PER = 25000
print("Loading datasets (shared for all models)...")

ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
add_from_ds(ds, "text", MAX_PER)

ds = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True)
add_from_ds(ds, "text", MAX_PER)

ds = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
add_from_ds(ds, "text", 15000)

ds = load_dataset("transformersbook/codeparrot", split="train", streaming=True)
count = 0
for x in ds:
    if count >= 15000:
        break
    if "content" in x and x["content"] and isinstance(x["content"], str):
        train_texts.append(x["content"][:2048])
        count += 1

import random
random.seed(42)
random.shuffle(train_texts)
train_texts = train_texts[:100000]
print(f"Total examples: {len(train_texts)}")

# %% [markdown]
# ## 5. Fine-Tune Loop

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset

BASE_SAVE = "/tmp/finetuned"
os.makedirs(BASE_SAVE, exist_ok=True)

def fine_tune_one(base_model_id, target_repo, label):
    print(f"\n{'='*60}")
    print(f"=== Fine-tuning {label} ({base_model_id})")
    print(f"=== Target: {target_repo}")
    print(f"{'='*60}\n")
    gc.collect()
    torch.cuda.empty_cache()

    # Load tokenizer
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize
    def encode(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding=False)

    dataset = Dataset.from_dict({"text": train_texts[:50000]})
    dataset = dataset.map(encode, remove_columns=["text"], desc="Tokenizing")
    dataset = dataset.filter(lambda x: len(x["input_ids"]) > 10)

    # Load model
    print(f"Loading model {label}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA config
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    # GPT-2 uses different naming
    if "gpt2" in base_model_id.lower():
        target_modules = ["c_attn", "c_proj", "c_fc"]

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Train
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    args = TrainingArguments(
        output_dir=f"{BASE_SAVE}/{target_repo}",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=2e-4,
        warmup_steps=100,
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

    # Save merged model
    save_dir = f"/tmp/{target_repo}"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    print(f"Saving merged model...")
    merged = model.merge_and_unload()
    merged.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)

    # Fix config
    cfg_path = os.path.join(save_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["_name_or_path"] = f"pinkelephantlimited/{target_repo}"
    cfg["architectures"] = ["LlamaForCausalLM" if "phi" in base_model_id.lower() else "GPT2LMHeadModel"]
    # GPT-2 is GPT2LMHeadModel, phi models use different architectures
    if "phi-4" in base_model_id.lower():
        cfg["architectures"] = ["Phi4ForCausalLM"]
    elif "phi-3" in base_model_id.lower():
        cfg["architectures"] = ["Phi3ForCausalLM"]
    elif "phi-2" in base_model_id.lower() or "phi-1" in base_model_id.lower():
        cfg["architectures"] = ["PhiForCausalLM"]
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    # Upload
    repo_id = f"pinkelephantlimited/{target_repo}"
    print(f"Uploading to {repo_id}...")
    api.create_repo(repo_id, private=False, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=save_dir, repo_id=repo_id, ignore_patterns=["*.bin"])
    print(f"✓ {label} done → https://huggingface.co/{repo_id}")

    # Cleanup
    del model, merged, trainer
    gc.collect()
    torch.cuda.empty_cache()

# %% [markdown]
# ## 6. Run

# %%
for base_id, target, label in MODELS:
    fine_tune_one(base_id, target, label)

print("\n✓ ALL MODELS FINISHED!")
