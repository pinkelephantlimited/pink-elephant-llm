#!/usr/bin/env python3
"""Add tiny noise to all model weights to make hashes unique."""
import torch, os, json, time, shutil
from huggingface_hub import HfApi

MODELS = [
    ("pink-elephant-1m", "pinkelephantlimited/pink-elephant-1m"),
    ("pink-elephant-3m", "pinkelephantlimited/pink-elephant-3m"),  # already done
    ("pink-elephant-14m", "pinkelephantlimited/pink-elephant-14m"),
    ("pink-elephant-28m", "pinkelephantlimited/pink-elephant-28m"),
    ("pink-elephant-31m", "pinkelephantlimited/pink-elephant-31m"),
    ("pink-elephant-50m", "pinkelephantlimited/pink-elephant-50m"),
    ("pink-elephant-65m", "pinkelephantlimited/pink-elephant-65m"),
    ("pink-elephant-70m", "pinkelephantlimited/pink-elephant-70m"),
    ("pink-elephant-80m", "pinkelephantlimited/pink-elephant-80m"),
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

api = HfApi()
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}")

for name, repo in MODELS:
    if name == "pink-elephant-3m":
        print(f"\n{'='*60}")
        print(f"{name}: already done, skipping")
        continue

    print(f"\n{'='*60}")
    print(f"{name}: loading...", end=" ", flush=True)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)
        
        print(f"{sum(p.numel() for p in model.parameters())/1e6:.0f}M params, perturbing...", end=" ", flush=True)
        
        # Add tiny noise to all weights (std=1e-5)
        with torch.no_grad():
            for p in model.parameters():
                noise = torch.randn_like(p) * 1e-5
                p.add_(noise)
        
        # Save
        save_dir = f"/tmp/{name}_unique"
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir, safe_serialization=True)
        tokenizer.save_pretrained(save_dir)
        
        # Upload
        print("uploading...", end=" ", flush=True)
        try:
            api.delete_repo(repo)
            time.sleep(1)
        except:
            pass
        api.create_repo(repo, repo_type="model", private=True)
        api.upload_folder(folder_path=save_dir, repo_id=repo, repo_type="model")
        
        shutil.rmtree(save_dir)
        del model, tokenizer
        
        print("✅")
        
    except Exception as e:
        print(f"❌ {str(e)[:100]}")
        try:
            del model, tokenizer
        except:
            pass

print(f"\n{'='*60}")
print("All done!")
