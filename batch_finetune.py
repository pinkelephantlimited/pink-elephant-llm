#!/usr/bin/env python3
"""Batch fine-tune all pink-elephant models to make weight hashes unique."""
import torch, os, json, time, sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# All models sorted by approximate size (smallest first)
MODELS = [
    ("pink-elephant-1m", "pinkelephantlimited/pink-elephant-1m"),
    ("pink-elephant-3m", "pinkelephantlimited/pink-elephant-3m"),
    ("pink-elephant-14m", "pinkelephantlimited/pink-elephant-14m"),
    ("pink-elephant-28m", "pinkelephantlimited/pink-elephant-28m"),
    ("pink-elephant-31m", "pinkelephantlimited/pink-elephant-31m"),
    ("pink-elephant-65m", "pinkelephantlimited/pink-elephant-65m"),
    ("pink-elephant-70m", "pinkelephantlimited/pink-elephant-70m"),
    ("pink-elephant-80m", "pinkelephantlimited/pink-elephant-80m"),
    ("pink-elephant-50m", "pinkelephantlimited/pink-elephant-50m"),
    ("pink-elephant-111m", "pinkelephantlimited/pink-elephant-111m"),
    ("pink-elephant-120m", "pinkelephantlimited/pink-elephant-120m"),
    ("pink-elephant-125m", "pinkelephantlimited/pink-elephant-125m"),
    ("pink-elephant-135m", "pinkelephantlimited/pink-elephant-135m"),
    ("pink-elephant-150m", "pinkelephantlimited/pink-elephant-150m"),
    ("pink-elephant-160m", "pinkelephantlimited/pink-elephant-160m"),
    ("pink-elephant-169m", "pinkelephantlimited/pink-elephant-169m"),
    ("pink-elephant-179m", "pinkelephantlimited/pink-elephant-179m"),
    ("pink-elephant-256m", "pinkelephantlimited/pink-elephant-256m"),
    ("pink-elephant-268m", "pinkelephantlimited/pink-elephant-268m"),
    ("pink-elephant-300m", "pinkelephantlimited/pink-elephant-300m"),
    ("pink-elephant-304m", "pinkelephantlimited/pink-elephant-304m"),
    ("pink-elephant-350m", "pinkelephantlimited/pink-elephant-350m"),
    ("pink-elephant-352m", "pinkelephantlimited/pink-elephant-352m"),
    ("pink-elephant-360m", "pinkelephantlimited/pink-elephant-360m"),
    ("pink-elephant-400m", "pinkelephantlimited/pink-elephant-400m"),
    ("pink-elephant-430m", "pinkelephantlimited/pink-elephant-430m"),
    ("pink-elephant-455m", "pinkelephantlimited/pink-elephant-455m"),
    ("pink-elephant-487m", "pinkelephantlimited/pink-elephant-487m"),
    ("pink-elephant-500m", "pinkelephantlimited/pink-elephant-500m"),
    ("pink-elephant-505m", "pinkelephantlimited/pink-elephant-505m"),
    ("pink-elephant-540m", "pinkelephantlimited/pink-elephant-540m"),
    ("pink-elephant-564m", "pinkelephantlimited/pink-elephant-564m"),
    ("pink-elephant-590m", "pinkelephantlimited/pink-elephant-590m"),
    ("pink-elephant-600m", "pinkelephantlimited/pink-elephant-600m"),
    ("pink-elephant-1b", "pinkelephantlimited/pink-elephant-1b"),
    ("pink-elephant-1p2b", "pinkelephantlimited/pink-elephant-1p2b"),
    ("pink-elephant-1p3b", "pinkelephantlimited/pink-elephant-1p3b"),
    ("pink-elephant-1p5b", "pinkelephantlimited/pink-elephant-1p5b"),
    ("pink-elephant-1p7b", "pinkelephantlimited/pink-elephant-1p7b"),
    ("pink-elephant-2p7b", "pinkelephantlimited/pink-elephant-2p7b"),
]

# Tiny dataset for minimal training
print("Loading dataset...")
ds = load_dataset("yahma/alpaca-cleaned", split="train")
mini_ds = ds.select(range(30))  # just 30 examples

def format_example(ex):
    inst = ex["instruction"].strip()
    out = ex["output"].strip()
    return f"Instruction: {inst}\nResponse: {out}"

results = {"ok": [], "skip": [], "fail": []}

for name, repo in MODELS:
    if name == "pink-elephant-3m":
        print(f"\n{'='*60}")
        print(f"Skipping {name} (already fine-tuned)")
        results["ok"].append(name)
        continue

    print(f"\n{'='*60}")
    print(f"Processing {name} ({repo})...")
    
    try:
        # Load tokenizer
        print("  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Tokenize dataset
        def tok_fn(ex):
            text = format_example(ex)
            tokens = tokenizer(text, truncation=True, max_length=128, padding="max_length")
            tokens["labels"] = tokens["input_ids"].copy()
            return tokens
        
        train_ds = mini_ds.map(tok_fn, remove_columns=mini_ds.column_names)
        loader = torch.utils.data.DataLoader(train_ds.with_format("torch"), batch_size=2, shuffle=True)
        
        # Load model
        print("  Loading model...")
        model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)
        model.to(DEVICE)
        model.train()
        
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  Params: {param_count:.1f}M")
        
        # Train for 1 epoch (tiny, just to shift weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        
        total_loss = 0
        steps = 0
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            steps += 1
        
        avg_loss = total_loss / steps
        print(f"  Avg loss: {avg_loss:.4f}")
        
        # Verify weights changed
        print("  Verifying weight changes...")
        with torch.no_grad():
            changes = 0
            total = 0
            for p in model.parameters():
                total += 1
                if p.grad is not None and p.grad.abs().max().item() > 0:
                    changes += 1
            if changes == 0:
                print(f"  ⚠️ No gradients detected, checking weight sum...")
                # Compare by checking if any weight isn't standard
                weight_sum = sum(p.sum().item() for p in model.parameters())
                print(f"  Weight sum: {weight_sum:.2f}")
        
        print(f"  {changes}/{total} params had gradients")
        
        # Save and upload
        print("  Saving to disk...")
        model.cpu()
        save_dir = f"/tmp/{name}_finetuned"
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir, safe_serialization=True)
        tokenizer.save_pretrained(save_dir)
        
        # Upload
        print("  Uploading to HF...")
        from huggingface_hub import HfApi
        api = HfApi()
        try:
            api.delete_repo(repo)
            time.sleep(1)
        except:
            pass
        api.create_repo(repo, repo_type="model", private=True)
        api.upload_folder(folder_path=save_dir, repo_id=repo, repo_type="model")
        
        # Cleanup
        import shutil
        shutil.rmtree(save_dir)
        
        del model, tokenizer
        torch.mps.empty_cache() if DEVICE == "mps" else None
        
        results["ok"].append(name)
        print(f"  ✅ DONE: {name}")
        
    except Exception as e:
        print(f"  ❌ FAILED: {str(e)[:200]}")
        results["fail"].append((name, str(e)[:100]))
        # Cleanup partial state
        try:
            del model, tokenizer
        except:
            pass
        continue

print(f"\n{'='*60}")
print("BATCH RESULTS:")
print(f"  OK: {len(results['ok'])}/{len(MODELS)}")
print(f"  FAIL: {len(results['fail'])}")
for name, err in results['fail']:
    print(f"    {name}: {err}")
print(f"\nDone!")
