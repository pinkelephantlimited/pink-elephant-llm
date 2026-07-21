#!/usr/bin/env python3
"""Fine-tune pink-elephant-3m to make weight hashes unique."""
import torch, os, json, time
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset

MODEL = "pinkelephantlimited/pink-elephant-3m"
OUTPUT = "pink-elephant-3m-finetuned"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Load model & tokenizer
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True)
model.to(DEVICE)
model.train()
tokenizer.pad_token = tokenizer.eos_token
print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# Load tiny dataset (100 examples from Alpaca)
print("Loading dataset...")
ds = load_dataset("yahma/alpaca-cleaned", split="train")
ds = ds.select(range(200))  # just 200 examples
print(f"Training examples: {len(ds)}")

def format_example(ex):
    inst = ex["instruction"].strip()
    inp = ex.get("input", "").strip()
    out = ex["output"].strip()
    if inp:
        text = f"Instruction: {inst}\nInput: {inp}\nResponse: {out}"
    else:
        text = f"Instruction: {inst}\nResponse: {out}"
    return text

# Tokenize
def tokenize_fn(ex):
    text = format_example(ex)
    tokens = tokenizer(text, truncation=True, max_length=256, padding="max_length")
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tok_ds = ds.map(tokenize_fn, remove_columns=ds.column_names)
loader = torch.utils.data.DataLoader(tok_ds.with_format("torch"), batch_size=4, shuffle=True)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=len(loader)*3)

# Train
print("Training...")
model.train()
global_step = 0
for epoch in range(3):
    total_loss = 0
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        global_step += 1
        if global_step % 20 == 0:
            print(f"  Step {global_step}, loss: {loss.item():.4f}")
    print(f"Epoch {epoch+1}, avg loss: {total_loss/len(loader):.4f}")

# Verify weights changed BEFORE upload
print("\nVerifying weight changes before upload...")
original = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True)
original.to("cpu")
model.to("cpu")

changes = 0
total = 0
max_diff = 0
for (n1, p1), (n2, p2) in zip(original.named_parameters(), model.named_parameters()):
    if n1 == n2:
        total += 1
        if not torch.equal(p1, p2):
            diff = (p1 - p2).abs().mean().item()
            max_diff = max(max_diff, diff)
            changes += 1
            if changes <= 3:
                print(f"  ✓ {n1}: mean diff = {diff:.6f}")

if changes == 0:
    print(f"  ❌ Weights IDENTICAL ({changes}/{total} changed)")
else:
    print(f"  ✅ {changes}/{total} parameter tensors changed (max diff: {max_diff:.6f})")

del original

# Save & upload
print("\nSaving...")
model.cpu()
model.save_pretrained(OUTPUT, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT)

print("Uploading to HF...")
from huggingface_hub import HfApi
api = HfApi()
repo_id = "pinkelephantlimited/pink-elephant-3m"
try:
    api.delete_repo(repo_id)
    time.sleep(2)
except:
    pass
api.create_repo(repo_id, repo_type="model", private=True)
api.upload_folder(folder_path=OUTPUT, repo_id=repo_id, repo_type="model")

print(f"\nDone! Model at: https://huggingface.co/{repo_id}")
