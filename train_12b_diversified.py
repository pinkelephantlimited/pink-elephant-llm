# /// script
# dependencies = ["transformers", "datasets", "accelerate", "huggingface_hub", "bitsandbytes", "torch"]
# ///

# ---
# jupyter:
#   marimo:
#     name: "Train 12B LLM on molab (96GB RTX PRO 6000)"
# ---

# %% [markdown]
# # Train 12B General-Purpose LLM on molab
#
# **GPU**: NVIDIA RTX Pro 6000 Blackwell (96GB VRAM) — free on molab
# **Architecture**: LLaMA ~12.3B params
# **Data**: FineWeb-Edu + FineWeb + OpenWebMath + SmolLM (cosmopedia-v2) + CodeParrot + Nemotron-Legal + Investopedia
# **Precision**: BF16 + 8-bit Adam (bitsandbytes)
# **Checkpoints**: Auto-uploaded to HF every 500 steps — resume from any session
#
# **OUTPUT**: Trained model uploads to https://huggingface.co/pinkelephantlimited/pink-elephant-12b
# All sources are verified working (Parquet format, no gating, no missing configs).

# %% [markdown]
# ## 1. Install Dependencies

# %%
import subprocess, sys
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "torch", "transformers", "datasets", "accelerate",
    "huggingface_hub", "bitsandbytes", "sentencepiece"
])
print("Installed!")

# %% [markdown]
# ## 2. Login to Hugging Face

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
# ## 3. Load Data (all verified Parquet sources)

# %%
from datasets import load_dataset

MAX_PER_SOURCE = 100000
MAX_TOTAL = 500000
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

# 1. General: FineWeb-Edu (SAT for sure)
print("=== 1. FineWeb-Edu ===")
try:
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                      split="train", streaming=True)
    c = add_from_ds(ds, "text", 80000)
    print(f"  {c} loaded ({len(train_texts)} total)")
except Exception as e:
    print(f"  Error: {e}")

# 2. General: FineWeb (SAT for sure)
print("=== 2. FineWeb ===")
try:
    ds = load_dataset("HuggingFaceFW/fineweb", "sample-10BT",
                      split="train", streaming=True)
    c = add_from_ds(ds, "text", 80000)
    print(f"  {c} loaded ({len(train_texts)} total)")
except Exception as e:
    print(f"  Error: {e}")

# 3. Math: OpenWebMath (SAT for sure)
print("=== 3. OpenWebMath ===")
try:
    ds = load_dataset("open-web-math/open-web-math", split="train",
                      streaming=True)
    c = add_from_ds(ds, "text", 50000)
    print(f"  {c} loaded ({len(train_texts)} total)")
except Exception as e:
    print(f"  Error: {e}")

# 4. Books: SmolLM (cosmopedia-v2)
print("=== 4. SmolLM Corpus (cosmopedia-v2) ===")
try:
    ds = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)
    count = 0
    for x in ds:
        if count >= 50000 or len(train_texts) >= MAX_TOTAL:
            break
        for f in ["text", "content"]:
            if f in x and x[f] and isinstance(x[f], str):
                train_texts.append(x[f][:2048])
                count += 1
                break
    print(f"  SmolLM: {count} loaded ({len(train_texts)} total)")
except Exception as e:
    print(f"  Error: {e}")

# 5. Code: codeparrot (mixed languages)
print("=== 5. CodeParrot ===")
try:
    ds = load_dataset("transformersbook/codeparrot", split="train", streaming=True)
    count = 0
    for x in ds:
        if count >= 50000 or len(train_texts) >= MAX_TOTAL:
            break
        if "content" in x and x["content"] and isinstance(x["content"], str):
            train_texts.append(x["content"][:2048])
            count += 1
    print(f"  CodeParrot: {count} loaded ({len(train_texts)} total)")
except Exception as e:
    print(f"  Error: {e}")

# 6. Legal: Nemotron
print("=== 6. Nemotron-Legal ===")
try:
    ds = load_dataset("nvidia/Nemotron-Pretraining-Legal-v1",
                      "Nemotron-Pretraining-Legal-Case-Law-Summary",
                      split="train", streaming=True)
    count = 0
    for x in ds:
        if count >= 40000 or len(train_texts) >= MAX_TOTAL:
            break
        for f in ["text", "input", "content"]:
            if f in x and x[f] and isinstance(x[f], str):
                train_texts.append(x[f][:2048])
                count += 1
                break
    print(f"  {count} loaded ({len(train_texts)} total)")
except Exception as e:
    print(f"  Error: {e}")

# 7. Finance: Investopedia (SAT for sure)
print("=== 7. Investopedia ===")
try:
    ds = load_dataset("infCapital/investopedia_terms_en", split="train",
                      streaming=True)
    c = add_from_ds(ds, "text", 20000)
    print(f"  {c} loaded ({len(train_texts)} total)")
except Exception as e:
    print(f"  Error: {e}")

print(f"\n=== TOTAL: {len(train_texts)} examples ===")

# %% [markdown]
# ## 4. Train Tokenizer (vocab=4096)

# %%
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast

tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=4096,
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
# ## 5. Create Model (~12.3B params)
#
# VRAM estimate: 25GB (weights) + 25GB (8bit Adam) + 25GB (grads) + ~12GB (acts @ batch=4, seq=4096) = ~87GB
# 96GB GPU has ~9GB headroom.

# %%
from transformers import LlamaConfig, LlamaForCausalLM

config = LlamaConfig(
    vocab_size=4096,
    hidden_size=5120,
    intermediate_size=13824,
    num_hidden_layers=40,
    num_attention_heads=40,
    max_position_embeddings=4096,
    rope_theta=10000.0,
    tie_word_embeddings=True,
    torch_dtype=torch.bfloat16,
)

model = LlamaForCausalLM(config)
total = sum(p.numel() for p in model.parameters())
print(f"Model: {total:,} params ({total/1e9:.2f}B)")
print(f"VRAM est: {total * 2 / 1e9:.1f}GB (weights) + {total * 2 / 1e9:.1f}GB (optim) + acts")

# %% [markdown]
# ## 6. Tokenize Dataset

# %%
from datasets import Dataset

MAX_LENGTH = 4096
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
# ## 7. Data Collator

# %%
from transformers import DataCollatorForLanguageModeling

collator = DataCollatorForLanguageModeling(
    tokenizer=hf_tokenizer, mlm=False, pad_to_multiple_of=8
)

# %% [markdown]
# ## 8. Train (12hrs on molab)
#
# batch=4, grad_accum=8 → effective 32
# bf16 + 8bit Adam + gradient checkpointing
# Saves every 500 steps and uploads to HF immediately.
# Full 4096 context — ~87 GB VRAM, 9 GB headroom

# %%
from transformers import TrainingArguments, Trainer, TrainerCallback
from huggingface_hub import HfApi

MODEL_NAME = "pink-elephant-12b"
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
                ignore_patterns=["*.bin", "optimizer.pt", "scheduler.pt", "rng_state.pth"],
            )

args = TrainingArguments(
    output_dir="./" + MODEL_NAME,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
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
    print(f"Resuming from: {resume}")

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
# ## 10. Upload to Hugging Face

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
api.upload_folder(folder_path=save_dir, repo_id=REPO_ID,
                  ignore_patterns=["*.bin"])
print(f"Uploaded: https://huggingface.co/{REPO_ID}")

# %%
print("DONE! Model at: https://huggingface.co/pinkelephantlimited/pink-elephant-12b")
