# /// script
# dependencies = ["transformers", "datasets", "accelerate", "huggingface_hub", "bitsandbytes", "torch"]
# ///

# ---
# jupyter:
#   marimo:
#     name: "Train 33M General-Purpose LLM"
# ---

# %% [markdown]
# # Train 33M General-Purpose LLM
#
# **GPU**: T4 (free Colab) — trains in ~2-3 hours
# **Architecture**: LLaMA ~33M params
# **Data**: FineWeb-Edu (general web, no domain specialization)
# **Precision**: BF16 + 8-bit Adam (bitsandbytes)

# %%
import subprocess, sys
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "torch", "transformers", "datasets", "accelerate",
    "huggingface_hub", "bitsandbytes", "sentencepiece"
])
print("Installed!")

# %%
import os, json, shutil, time, glob, random
import torch
import bitsandbytes as bnb

# %%
from huggingface_hub import login
import getpass
token = getpass.getpass("HF token: ")
login(token, add_to_git_credential=False)
print("Logged in!")

# %%
from datasets import load_dataset

MAX_EXAMPLES = 150000
train_texts = []

print("=== General: FineWeb-Edu ===")
try:
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                      split="train", streaming=True)
    count = 0
    for x in ds:
        if count >= MAX_EXAMPLES:
            break
        if "text" in x and x["text"]:
            train_texts.append(x["text"][:1024])
            count += 1
    print(f"  {count} loaded")
except Exception as e:
    print(f"  Error: {e}")

print(f"\n=== TOTAL: {len(train_texts)} examples ===")

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
print(f"Test: {hf_tokenizer.decode(hf_tokenizer.encode('The court finds that the defendant'))}")

# %%
from transformers import LlamaConfig, LlamaForCausalLM

config = LlamaConfig(
    vocab_size=hf_tokenizer.vocab_size,
    hidden_size=512,
    intermediate_size=1792,
    num_hidden_layers=8,
    num_attention_heads=8,
    max_position_embeddings=2048,
    rope_theta=10000.0,
    tie_word_embeddings=True,
    torch_dtype=torch.bfloat16,
)

model = LlamaForCausalLM(config)
total = sum(p.numel() for p in model.parameters())
print(f"Model: {total:,} params ({total/1e6:.1f}M)")

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

# %%
from transformers import DataCollatorForLanguageModeling

collator = DataCollatorForLanguageModeling(
    tokenizer=hf_tokenizer, mlm=False, pad_to_multiple_of=8
)

# %%
from transformers import TrainingArguments, Trainer, TrainerCallback
from huggingface_hub import HfApi

MODEL_NAME = "pink-elephant-33m"
REPO_ID = f"pinkelephantlimited/{MODEL_NAME}"

class HFSaveCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        ckpt_dir = f"{args.output_dir}/checkpoint-{state.global_step}"
        if os.path.exists(ckpt_dir):
            print(f"\nUploading checkpoint-{state.global_step} to HF...")
            try:
                api = HfApi()
                api.upload_folder(
                    folder_path=ckpt_dir,
                    repo_id=REPO_ID,
                    path_in_repo=f"checkpoints/checkpoint-{state.global_step}",
                    ignore_patterns=["*.bin", "optimizer.pt", "scheduler.pt", "rng_state.pth"],
                )
                print(f"  -> Uploaded to {REPO_ID}")
            except Exception as e:
                print(f"  -> Upload failed: {e}")

args = TrainingArguments(
    output_dir="./" + MODEL_NAME,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    learning_rate=3e-4,
    weight_decay=0.01,
    warmup_steps=200,
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

# %%
prompts = [
    "The definition of artificial intelligence is",
    "In the beginning,",
    "The capital of France is",
    "Machine learning is a",
    "The most important invention is",
]
for p in prompts:
    inputs = hf_tokenizer(p, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=60, do_sample=True,
            temperature=0.5, top_p=0.9,
            pad_token_id=hf_tokenizer.pad_token_id,
        )
    print(f"\n{p}\n  -> {hf_tokenizer.decode(out[0], skip_special_tokens=True)}")

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

api = HfApi()
api.upload_folder(folder_path=save_dir, repo_id=REPO_ID,
                  ignore_patterns=["*.bin"])
print(f"Uploaded: https://huggingface.co/{REPO_ID}")

# %%
print("DONE! Model at: https://huggingface.co/pinkelephantlimited/pink-elephant-33m")
