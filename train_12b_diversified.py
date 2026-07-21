# /// script
# dependencies = ["transformers", "datasets", "accelerate", "huggingface_hub", "bitsandbytes", "torch"]
# ///

# ---
# jupyter:
#   marimo:
#     name: "Train 12B Diversified LLM on molab (96GB RTX PRO 6000)"
# ---

# %% [markdown]
# # Train 12B Diversified LLM on molab
#
# **GPU**: NVIDIA RTX Pro 6000 Blackwell (96GB VRAM) — free on molab
# **Architecture**: LLaMA ~12.3B params
# **Data**: General + Legal + Finance + Code (diversified)
# **Precision**: BF16 + 8-bit Adam (bitsandbytes)
#
# ## How to use on molab
# 1. Go to https://molab.marimo.io
# 2. Click "New notebook" → "Blank"
# 3. Copy each cell below into its own marimo cell
# 4. Toggle GPU: Click notebook specs → Attach GPU
# 5. Run all cells

# %% [markdown]
# ## 1. Install & Login

# %%
import subprocess, sys, os, json, shutil, time, glob, random
import torch
import bitsandbytes as bnb

subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers", "datasets", "accelerate",
    "huggingface_hub", "bitsandbytes", "sentencepiece"
], capture_output=True)

# %%
from huggingface_hub import login
import getpass
token = getpass.getpass("HF token: ")
login(token, add_to_git_credential=False)
print("Logged in!")

# %% [markdown]
# ## 2. Load Diversified Data (~50-100M tokens)

# %%
from datasets import load_dataset

MAX_PER_SOURCE = 120000
MAX_TOTAL = 500000
train_texts = []

# --- General (35%) ---
print("=== General: FineWeb-Edu ===")
try:
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                      split="train", streaming=True)
    count = 0
    for x in ds:
        if count >= MAX_PER_SOURCE or len(train_texts) >= MAX_TOTAL:
            break
        if "text" in x and x["text"]:
            train_texts.append(x["text"][:2048])
            count += 1
    print(f"  {count} loaded ({len(train_texts)} total)")
except Exception as e:
    print(f"  Error: {e}")

# --- Legal (20%) ---
print("=== Legal: Nemotron ===")
legal_configs = [
    "Nemotron-Pretraining-Legal-California-Code-Of-Regulations",
    "Nemotron-Pretraining-Legal-Case-Law-Summary",
    "Nemotron-Pretraining-Legal-eCFR",
    "Nemotron-Pretraining-Legal-GlobalCit",
    "Nemotron-Pretraining-Legal-CaseHOLD",
]
for cfg in legal_configs:
    try:
        ds = load_dataset("nvidia/Nemotron-Pretraining-Legal-v1", cfg,
                          split="train", streaming=True)
        count = 0
        for x in ds:
            if count >= 30000 or len(train_texts) >= MAX_TOTAL:
                break
            for f in ["text", "input", "content"]:
                if f in x and x[f] and isinstance(x[f], str):
                    train_texts.append(x[f][:2048])
                    count += 1
                    break
        print(f"  {cfg}: {count}")
    except Exception as e:
        print(f"  Skip {cfg}: {e}")

# --- Finance (20%) ---
print("=== Finance: SEC Reports ===")
try:
    ds = load_dataset("JanosAudran/financial-reports-sec", split="train",
                      streaming=True)
    count = 0
    for x in ds:
        if count >= MAX_PER_SOURCE or len(train_texts) >= MAX_TOTAL:
            break
        if "text" in x and x["text"]:
            train_texts.append(x["text"][:2048])
            count += 1
    print(f"  SEC: {count}")
except Exception as e:
    print(f"  Error: {e}")

# --- Code (25%) ---
print("=== Code: The Stack ===")
try:
    ds = load_dataset("bigcode/the-stack-smol", split="train", streaming=True)
    count = 0
    for x in ds:
        if count >= MAX_PER_SOURCE or len(train_texts) >= MAX_TOTAL:
            break
        if "content" in x and x["content"]:
            train_texts.append(x["content"][:2048])
            count += 1
    print(f"  Code: {count}")
except Exception as e:
    print(f"  Error: {e}")

print(f"\n=== TOTAL: {len(train_texts)} examples ===")

# %% [markdown]
# ## 3. Train Tokenizer (vocab=16384)

# %%
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast

tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=16384,
    special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
    min_frequency=2,
)
tokenizer.train_from_iterator(train_texts, trainer)

hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="<unk>",
    bos_token="<s>",
    eos_token="</s>",
    pad_token="<pad>",
)
print(f"Vocab: {hf_tokenizer.vocab_size}")
print(f"Test: {hf_tokenizer.decode(hf_tokenizer.encode('The court finds that the defendant'))}")

# %% [markdown]
# ## 4. Create Model (~12.3B params)
#
# | Component | Value |
# |---|---|
# | Vocab | 16,384 |
# | Hidden | 4,608 |
# | Layers | 36 |
# | Heads | 32 |
# | Intermediate | 18,432 |
# | Params | ~12.3B |
# | VRAM est | ~78GB (bf16 + 8-bit Adam) |

# %%
from transformers import LlamaConfig, LlamaForCausalLM

config = LlamaConfig(
    vocab_size=hf_tokenizer.vocab_size,
    hidden_size=4608,
    intermediate_size=18432,
    num_hidden_layers=36,
    num_attention_heads=32,
    max_position_embeddings=2048,
    rope_theta=10000.0,
    tie_word_embeddings=True,
    torch_dtype=torch.bfloat16,
)

model = LlamaForCausalLM(config)
total = sum(p.numel() for p in model.parameters())
print(f"Model: {total:,} params ({total/1e9:.2f}B)")
print(f"VRAM (bf16 weights): {total * 2 / 1e9:.1f}GB")
print(f"VRAM (8-bit optim): {total * 2 / 1e9:.1f}GB")
print(f"Total est: {total * 6 / 1e9:.1f}GB + activations")

# %% [markdown]
# ## 5. Tokenize Dataset

# %%
from datasets import Dataset

MAX_LENGTH = 512

random.seed(42)
random.shuffle(train_texts)

def encode(texts):
    return hf_tokenizer(texts, truncation=True, max_length=MAX_LENGTH,
                        padding=False)["input_ids"]

dataset = Dataset.from_dict({"text": train_texts})
dataset = dataset.map(
    lambda x: {"input_ids": encode(x["text"])},
    remove_columns=["text"], desc="Tokenizing",
)
dataset = dataset.filter(lambda x: len(x["input_ids"]) > 10, desc="Filtering")
print(f"Examples: {len(dataset)}")

# %% [markdown]
# ## 6. Data Collator

# %%
from transformers import DataCollatorForLanguageModeling

collator = DataCollatorForLanguageModeling(
    tokenizer=hf_tokenizer, mlm=False, pad_to_multiple_of=8
)

# %% [markdown]
# ## 7. Train (12hrs on molab)
#
# **Settings**:
# - Batch: 1 per device (large model), 32 grad accum = 32 effective
# - Precision: bf16 mixed
# - Optimizer: 8-bit Adam (bitsandbytes)
# - Grad checkpointing: ON
# - LR: 2e-4 cosine
#
# Checkpoints save to /tmp/ on molab — download before session ends.

# %%
from transformers import TrainingArguments, Trainer

MODEL_NAME = "pink-elephant-12b"

args = TrainingArguments(
    output_dir="./" + MODEL_NAME,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    num_train_epochs=10,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=500,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
    bf16=True,
    optim="adamw_8bit",
    dataloader_num_workers=2,
    remove_unused_columns=False,
    gradient_checkpointing=True,
    ddp_find_unused_parameters=False,
)

# Find latest checkpoint for resume
resume = None
ckpts = sorted(glob.glob(f"./{MODEL_NAME}/checkpoint-*"))
if ckpts:
    resume = ckpts[-1]
    print(f"Resuming from: {resume}")

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=collator,
)
trainer.train(resume_from_checkpoint=resume)

# %% [markdown]
# ## 8. Quick Test

# %%
prompts = [
    "The court finds that",
    "In accordance with IFRS standards,",
    "The function computes",
    "According to the financial statements,",
    "The auditor shall",
]
for p in prompts:
    inputs = hf_tokenizer(p, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=80, do_sample=True,
            temperature=0.5, top_p=0.9,
            pad_token_id=hf_tokenizer.pad_token_id,
        )
    print(f"\n{p}\n  -> {hf_tokenizer.decode(out[0], skip_special_tokens=True)}")

# %% [markdown]
# ## 9. Upload to Hugging Face

# %%
from huggingface_hub import HfApi

save_dir = "/tmp/" + MODEL_NAME
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)

model.save_pretrained(save_dir, safe_serialization=True)
hf_tokenizer.save_pretrained(save_dir)

cfg_path = os.path.join(save_dir, "config.json")
with open(cfg_path) as f:
    cfg = json.load(f)
cfg["_name_or_path"] = f"pinkelephantlimited/{MODEL_NAME}"
cfg["architectures"] = ["LlamaForCausalLM"]
with open(cfg_path, "w") as f:
    json.dump(cfg, f, indent=2)

license_text = """MIT License
Copyright (c) 2026 Pink Elephant Limited
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""
with open(os.path.join(save_dir, "LICENSE"), "w") as f:
    f.write(license_text)

api = HfApi()
repo_id = f"pinkelephantlimited/{MODEL_NAME}"
api.upload_folder(folder_path=save_dir, repo_id=repo_id,
                  ignore_patterns=["*.bin"])
print(f"Uploaded: https://huggingface.co/{repo_id}")

# %%
print("DONE! Model at: https://huggingface.co/pinkelephantlimited/pink-elephant-12b")
