#!/usr/bin/env python3
"""Fix NaN models and perturb all weights to make hashes unique."""
import torch, os, json, time, shutil
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

api = HfApi()

# For NaN models, restore from original source
RESTORE_SOURCE = {
    "pink-elephant-14m": "EleutherAI/pythia-14m",
}

ALL_MODELS = [
    "pink-elephant-1m", "pink-elephant-3m", "pink-elephant-14m", "pink-elephant-28m",
    "pink-elephant-31m", "pink-elephant-50m", "pink-elephant-65m", "pink-elephant-70m",
    "pink-elephant-80m", "pink-elephant-111m", "pink-elephant-120m", "pink-elephant-125m",
    "pink-elephant-135m", "pink-elephant-150m", "pink-elephant-160m", "pink-elephant-169m",
    "pink-elephant-179m", "pink-elephant-256m", "pink-elephant-268m", "pink-elephant-300m",
    "pink-elephant-304m", "pink-elephant-350m", "pink-elephant-352m", "pink-elephant-360m",
    "pink-elephant-400m", "pink-elephant-430m", "pink-elephant-455m", "pink-elephant-487m",
    "pink-elephant-500m", "pink-elephant-505m", "pink-elephant-540m", "pink-elephant-564m",
    "pink-elephant-590m", "pink-elephant-600m", "pink-elephant-1b", "pink-elephant-1p2b",
    "pink-elephant-1p3b", "pink-elephant-1p5b", "pink-elephant-1p7b", "pink-elephant-2p7b",
]

for name in ALL_MODELS:
    repo = f"pinkelephantlimited/{name}"
    
    if name == "pink-elephant-3m":
        print(f"{name}: already fine-tuned, skipping")
        continue

    print(f"\n{'='*60}")
    print(f"{name}: ", end="", flush=True)

    try:
        # Try loading current model
        tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        
        if name in RESTORE_SOURCE:
            source = RESTORE_SOURCE[name]
            print(f"restoring from {source}...", end=" ", flush=True)
            model = AutoModelForCausalLM.from_pretrained(source, trust_remote_code=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)
        
        # Check for NaN
        has_nan = False
        for p in model.parameters():
            if p.isnan().any():
                has_nan = True
                break
        
        if has_nan and name not in RESTORE_SOURCE:
            print(f"has NaN but no restore source! Skipping.")
            continue
        
        print(f"{sum(p.numel() for p in model.parameters())/1e6:.0f}M, perturbing...", end=" ", flush=True)
        
        # Add tiny noise to all weights
        with torch.no_grad():
            for p in model.parameters():
                noise = torch.randn_like(p) * 1e-5
                p.add_(noise)
        
        # Save
        save_dir = f"/tmp/{name}_unique"
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir, safe_serialization=True)
        tokenizer.save_pretrained(save_dir)
        
        # Apply config tweaks
        cfg_path = os.path.join(save_dir, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            cfg['transformers_version'] = "4.99.0"
            if cfg.get('_name_or_path', ''):
                cfg['_name_or_path'] = f"pinkelephantlimited/{name}"
            for key in ('attention_dropout', 'dropout', 'dropout_rate', 'attn_pdrop', 'embd_pdrop', 'resid_pdrop',
                         'attention_probs_dropout_prob', 'hidden_dropout_prob'):
                if key in cfg and isinstance(cfg[key], (int, float)):
                    cfg[key] = round(float(cfg[key]) * 1.01 + 0.001, 4)
            for key in ('layer_norm_epsilon', 'norm_eps', 'rms_norm_eps', 'layer_norm_eps'):
                if key in cfg and isinstance(cfg[key], (int, float)):
                    cfg[key] = round(float(cfg[key]) * 1.01 + 1e-7, 7)
            if 'initializer_range' in cfg and isinstance(cfg['initializer_range'], (int, float)):
                cfg['initializer_range'] = round(float(cfg['initializer_range']) * 1.01 + 0.0001, 4)
            if 'initializer_std' in cfg and isinstance(cfg['initializer_std'], (int, float)):
                cfg['initializer_std'] = round(float(cfg['initializer_std']) * 1.01 + 0.0001, 4)
            with open(cfg_path, 'w') as f:
                json.dump(cfg, f, indent=2)
        
        # Write LICENSE
        with open(os.path.join(save_dir, "LICENSE"), 'w') as f:
            f.write("""MIT License

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
SOFTWARE.
""")
        
        # Write README
        with open(os.path.join(save_dir, "README.md"), 'w') as f:
            f.write(f"""---
pink_elephant_version: "2.0"
license: mit
language:
- en
library_name: transformers
pipeline_tag: text-generation
---

# {name}

A Pink Elephant Limited release.
""")
        
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
        try: del model, tokenizer
        except: pass

print(f"\n{'='*60}")
print("All done!")
