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
# **Data**: FineWeb-Edu + FineWeb + OpenWebMath + Nemotron-Legal + Investopedia
# **Precision**: BF16 + 8-bit Adam (bitsandbytes)
#
# **OUTPUT**: Trained model uploads to https://huggingface.co/pinkelephantlimited/pink-elephant-12b
# All 7 sources are verified working (Parquet format, no gating, no missing configs).

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
import os, json, shutil, time, glob, random
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

MAX_PER_SOURCE = 80000
MAX_TOTAL = 400000
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

# 4. Legal: Nemotron (SAT for sure)
print("=== 4. Nemotron-Legal ===")
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

# 5. Finance: Investopedia (SAT for sure)
print("=== 5. Investopedia ===")
try:
    ds = load_dataset("infCapital/investopedia_terms_en", split="train",
                      streaming=True)
    c = add_from_ds(ds, "text", 20000)
    print(f"  {c} loaded ({len(train_texts)} total)")
except Exception as e:
    print(f"  Error: {e}")

print(f"\n=== TOTAL: {len(train_texts)} examples ===")

# %% [markdown]
# ## 4. Train Tokenizer (vocab=16384)

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
# ## 5. Create Model (~12.3B params)
#
# VRAM estimate: 24GB (bf16 weights) + 24GB (8bit Adam) + 12GB (grads) + ~8GB (acts) = ~68GB
# 96GB GPU has ~28GB headroom. Could potentially increase batch size.

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
print(f"VRAM est: {total * 2 / 1e9:.1f}GB (weights) + {total * 2 / 1e9:.1f}GB (optim) + acts")

# %% [markdown]
# ## 6. Tokenize Dataset

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
# ## 7. Data Collator

# %%
from transformers import DataCollatorForLanguageModeling

collator = DataCollatorForLanguageModeling(
    tokenizer=hf_tokenizer, mlm=False, pad_to_multiple_of=8
)

# %% [markdown]
# ## 8. Train (12hrs on molab)
#
# batch=1, grad_accum=32 → effective 32
# bf16 + 8bit Adam → ~68GB VRAM

# %%
from transformers import TrainingArguments, Trainer, TrainerCallback
from huggingface_hub import HfApi

MODEL_NAME = "pink-elephant-12b"
REPO_ID = f"pinkelephantlimited/{MODEL_NAME}"

# Ensure repo exists
HfApi().create_repo(REPO_ID, private=False, repo_type="model", exist_ok=True)
print(f"Repo {REPO_ID} ready")

class HFSaveCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = f"{args.output_dir}/checkpoint-{state.global_step}"
        if os.path.exists(ckpt_dir):
            print(f"\nUploading checkpoint-{state.global_step} to HF...")
            HfApi().upload_folder(
                folder_path=ckpt_dir,
                repo_id=REPO_ID,
                path_in_repo=f"checkpoints/checkpoint-{state.global_step}",
                ignore_patterns=["*.bin", "optimizer.pt", "scheduler.pt", "rng_state.pth"],
            )
            print(f"  -> Uploaded")

args = TrainingArguments(
    output_dir="./" + MODEL_NAME,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    num_train_epochs=10,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=500,
    logging_steps=10,
    save_strategy="steps",
    save_steps=1000,
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
