# /// script
# dependencies = ["transformers", "datasets", "accelerate", "huggingface_hub", "bitsandbytes", "torch"]
# ///

# ---
# jupyter:
#   marimo:
#     name: "Train 44M General LLM on molab"
# ---

# %% [markdown]
# # Train 44M General-Purpose LLM on molab
#
# **GPU**: NVIDIA RTX Pro 6000 Blackwell (96GB VRAM) — free on molab
# **Architecture**: LLaMA ~44M params (hidden=640, layers=6, heads=10)
# **Data**: FineWeb-Edu + FineWeb + OpenWebMath + SmolLM + CodeParrot + Nemotron-Legal + Investopedia
# **Precision**: BF16 + 8-bit Adam (bitsandbytes)
# **Checkpoints**: Auto-uploaded to HF every 500 steps — resume from any session

# %% [markdown]
# ## 1. Install

# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "torch", "transformers", "datasets", "accelerate",
    "huggingface_hub", "bitsandbytes", "sentencepiece"])
print("Installed!")

# %% [markdown]
# ## 2. Login to HF

# %%
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json, shutil, time, glob, random
import torch
import bitsandbytes as bnb
from huggingface_hub import login
import getpass
token = getpass.getpass("HF token: ")
login(token, add_to_git_credential=False)
print("Logged in!")

# %% [markdown]
# ## 3. Load Data (7 verified Parquet sources)

# %%
from datasets import load_dataset

MAX_TOTAL = 300000
train_texts = []

def add_from_ds(ds, key, limit, text_len=2048):
    count = 0
    for x in ds:
        if count >= limit or len(train_texts) >= MAX_TOTAL:
            break
        if key in x and x[key] and isinstance(x[key], str):
            train_texts.append(x[key][:text_len])
            count += 1
    return count

print("=== 1. FineWeb-Edu ===")
ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
print(f"  {add_from_ds(ds, 'text', 60000)} loaded ({len(train_texts)} total)")

print("=== 2. FineWeb ===")
ds = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True)
print(f"  {add_from_ds(ds, 'text', 60000)} loaded ({len(train_texts)} total)")

print("=== 3. OpenWebMath ===")
ds = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
print(f"  {add_from_ds(ds, 'text', 40000)} loaded ({len(train_texts)} total)")

print("=== 4. SmolLM Corpus ===")
ds = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)
count = 0
for x in ds:
    if count >= 40000 or len(train_texts) >= MAX_TOTAL:
        break
    for f in ["text", "content"]:
        if f in x and x[f] and isinstance(x[f], str):
            train_texts.append(x[f][:2048])
            count += 1
            break
print(f"  SmolLM: {count} loaded ({len(train_texts)} total)")

print("=== 5. CodeParrot ===")
ds = load_dataset("transformersbook/codeparrot", split="train", streaming=True)
count = 0
for x in ds:
    if count >= 30000 or len(train_texts) >= MAX_TOTAL:
        break
    if "content" in x and x["content"] and isinstance(x["content"], str):
        train_texts.append(x["content"][:2048])
        count += 1
print(f"  CodeParrot: {count} loaded ({len(train_texts)} total)")

print("=== 6. Nemotron-Legal ===")
ds = load_dataset("nvidia/Nemotron-Pretraining-Legal-v1",
                  "Nemotron-Pretraining-Legal-Case-Law-Summary",
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
print(f"  {count} loaded ({len(train_texts)} total)")

print("=== 7. Investopedia ===")
ds = load_dataset("infCapital/investopedia_terms_en", split="train", streaming=True)
print(f"  {add_from_ds(ds, 'text', 15000)} loaded ({len(train_texts)} total)")

print(f"\n=== TOTAL: {len(train_texts)} examples ===")

# %% [markdown]
# ## 4. Train Tokenizer

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

# %% [markdown]
# ## 5. Create Model (43.9M params)
#
# Architecture: hidden=640, intermediate=2048, layers=6, heads=10
# VRAM: ~88MB (weights) + ~88MB (8bit Adam) + ~88MB (grads) + ~50MB (acts) ≈ 314MB
# Fits comfortably on any GPU, even T4.

# %%
from transformers import LlamaConfig, LlamaForCausalLM

config = LlamaConfig(
    vocab_size=16384,
    hidden_size=640,
    intermediate_size=2048,
    num_hidden_layers=6,
    num_attention_heads=10,
    max_position_embeddings=1024,
    rope_theta=10000.0,
    tie_word_embeddings=True,
)
model = LlamaForCausalLM(config)
model = model.to(torch.bfloat16)
total = sum(p.numel() for p in model.parameters())
print(f"Model: {total:,} params ({total/1e6:.1f}M)")

# %% [markdown]
# ## 6. Tokenize Dataset

# %%
from datasets import Dataset

MAX_LENGTH = 1024
random.seed(42)
random.shuffle(train_texts)

def encode(texts):
    return hf_tokenizer(texts, truncation=True, max_length=MAX_LENGTH, padding=False)["input_ids"]

dataset = Dataset.from_dict({"text": train_texts})
dataset = dataset.map(lambda x: {"input_ids": encode(x["text"])}, remove_columns=["text"], desc="Tokenizing")
dataset = dataset.filter(lambda x: len(x["input_ids"]) > 10, desc="Filtering")
print(f"Examples: {len(dataset)}")

# %% [markdown]
# ## 7. Data Collator

# %%
from transformers import DataCollatorForLanguageModeling
collator = DataCollatorForLanguageModeling(tokenizer=hf_tokenizer, mlm=False, pad_to_multiple_of=8)

# %% [markdown]
# ## 8. Train
#
# Saves every 500 steps, uploads to HF.
# If session dies, re-open and it resumes from the latest checkpoint.

# %%
from transformers import TrainingArguments, Trainer, TrainerCallback
from huggingface_hub import HfApi

MODEL_NAME = "pink-elephant-44m"
REPO_ID = f"pinkelephantlimited/{MODEL_NAME}"

HfApi().create_repo(REPO_ID, private=False, repo_type="model", exist_ok=True)
print(f"Repo {REPO_ID} ready")

class HFSaveCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = f"{args.output_dir}/checkpoint-{state.global_step}"
        if os.path.exists(ckpt_dir):
            print(f"\nUploading checkpoint-{state.global_step}...")
            HfApi().upload_folder(
                folder_path=ckpt_dir,
                repo_id=REPO_ID,
                path_in_repo=f"checkpoints/checkpoint-{state.global_step}",
                ignore_patterns=["*.bin", "optimizer.pt"],
            )

args = TrainingArguments(
    output_dir="./" + MODEL_NAME,
    per_device_train_batch_size=128,
    gradient_accumulation_steps=1,
    num_train_epochs=10,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=500,
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    report_to="none",
    bf16=True,
    optim="adamw_8bit",
    dataloader_num_workers=2,
    remove_unused_columns=False,
    gradient_checkpointing=True,
    ddp_find_unused_parameters=False,
)

resume = None
ckpts = sorted(glob.glob(f"./{MODEL_NAME}/checkpoint-*"))
if ckpts:
    resume = ckpts[-1]
    print(f"Resuming from local checkpoint: {resume}")
else:
    print("No local checkpoints. Checking HF for remote checkpoints...")
    try:
        api = HfApi()
        files = api.list_repo_files(REPO_ID, repo_type="model")
        ckpt_dirs = set()
        for f in files:
            if f.startswith("checkpoints/checkpoint-"):
                if len(f.split("/")) >= 2:
                    ckpt_dirs.add(f.split("/")[1])
        if ckpt_dirs:
            latest = sorted(ckpt_dirs)[-1]
            dst_dir = f"./{MODEL_NAME}/{latest}"
            os.makedirs(dst_dir, exist_ok=True)
            print(f"Downloading remote checkpoint: {latest}")
            from huggingface_hub import hf_hub_download
            for fname in ["config.json", "generation_config.json", "model.safetensors",
                          "tokenizer.json", "tokenizer_config.json", "trainer_state.json"]:
                try:
                    path = hf_hub_download(repo_id=REPO_ID, filename=f"checkpoints/{latest}/{fname}", repo_type="model")
                    shutil.copy2(path, os.path.join(dst_dir, fname))
                    print(f"  Downloaded {fname}")
                except:
                    pass
            if os.path.exists(os.path.join(dst_dir, "trainer_state.json")):
                resume = dst_dir
                print(f"Resuming from remote checkpoint: {resume}")
            else:
                print("Checkpoint download incomplete — starting from scratch")
    except Exception as e:
        print(f"Error downloading checkpoint: {e}")

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=collator,
    callbacks=[HFSaveCallback],
)
trainer.train(resume_from_checkpoint=resume)

# %% [markdown]
# ## 9. Quick Test

# %%
prompts = [
    "The water cycle consists of",
    "Machine learning is a",
    "In accordance with the agreement,",
    "The function computes the",
    "According to the financial statements,",
]
for p in prompts:
    inputs = hf_tokenizer(p, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=60, do_sample=True,
            temperature=0.5, top_p=0.9,
            pad_token_id=hf_tokenizer.pad_token_id)
    print(f"\n{p}\n  -> {hf_tokenizer.decode(out[0], skip_special_tokens=True)}")

# %% [markdown]
# ## 10. Upload to HF

# %%
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
api = HfApi()
api.create_repo(REPO_ID, private=False, repo_type="model", exist_ok=True)
api.upload_folder(folder_path=save_dir, repo_id=REPO_ID, ignore_patterns=["*.bin"])
print(f"Uploaded: https://huggingface.co/{REPO_ID}")

# %%
print("DONE!")
