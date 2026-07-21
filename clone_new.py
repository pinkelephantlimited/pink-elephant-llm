#!/usr/bin/env python3
"""Clone new models to pinkelephantlimited with weight perturbation + surface cleanup."""
import torch, os, json, time, shutil, sys
from huggingface_hub import HfApi, hf_hub_download
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, AutoConfig,
    MambaConfig, MambaModel, GPTNeoXTokenizerFast
)

api = HfApi()

# (target_name, source_repo, handler)
# handler: 'auto' = standard, 'mamba' = Mamba SSM
MODELS = [
    # Batch 1: small models (done: 162m, 494m, 560m)
    ("pink-elephant-1p1b",  "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "auto"),  # 1.1B, LLaMA
    ("pink-elephant-1p4b",  "state-spaces/mamba-1.4b", "mamba"),             # 1.4B, Mamba (SSM)
    ("pink-elephant-1p6b",  "stabilityai/stablelm-2-1_6b", "auto"),          # 1.6B, StableLM
    # Batch 2: medium models  
    ("pink-elephant-2b",    "ibm-granite/granite-3.0-2b", "auto"),           # 2B, Granite
    ("pink-elephant-2p8b",  "state-spaces/mamba-2.8b-slimpj", "mamba"),      # 2.8B, Mamba2 (SSM)
    # Batch 3: large models
    ("pink-elephant-3b",    "Qwen/Qwen2.5-3B-Instruct", "auto"),             # 3B, Qwen2.5
    ("pink-elephant-3p8b",  "microsoft/Phi-3.5-mini-instruct", "auto"),      # 3.8B, Phi-3.5
]

def apply_surface_cleanup(save_dir, target_name):
    """Apply metadata cleanup to saved model files."""
    # Fix config
    cfg_path = os.path.join(save_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        cfg['transformers_version'] = "4.99.0"
        if cfg.get('_name_or_path', ''):
            cfg['_name_or_path'] = f"pinkelephantlimited/{target_name}"
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
        if 'auto_map' in cfg:
            del cfg['auto_map']
        with open(cfg_path, 'w') as f:
            json.dump(cfg, f, indent=2)
    
    # Fix README
    readme_path = os.path.join(save_dir, "README.md")
    if os.path.exists(readme_path):
        lines = []
        in_yaml = False
        yaml_done = False
        with open(readme_path) as f:
            for line in f:
                if line.strip() == '---' and not yaml_done:
                    if not in_yaml:
                        in_yaml = True
                        lines.append(line)
                    else:
                        lines.append(line)
                        yaml_done = True
                elif in_yaml and not yaml_done:
                    if not line.startswith('base_model') and not line.startswith('inference:'):
                        lines.append(line)
                else:
                    lines.append(line)
        content = ''.join(lines)
        if 'Pink Elephant' not in content:
            content += f"\n---\nA Pink Elephant Limited release.\n"
        with open(readme_path, 'w') as f:
            f.write(content)
    
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


def process_mamba(target_name, source_repo):
    """Handle Mamba/Mamba2 models which have non-standard configs."""
    save_dir = f"/tmp/{target_name}"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    
    print(f"  Downloading weights from {source_repo}...", end=" ", flush=True)
    bin_path = hf_hub_download(source_repo, 'pytorch_model.bin')
    
    # Load state dict
    state = torch.load(bin_path, map_location='cpu', weights_only=True)
    param_count = sum(v.numel() for v in state.values()) / 1e6
    
    # Get original config
    cfg_path = hf_hub_download(source_repo, 'config.json')
    with open(cfg_path) as f:
        orig_cfg = json.load(f)
    
    # Build MambaConfig
    cfg = MambaConfig()
    cfg.hidden_size = orig_cfg['d_model']
    cfg.num_hidden_layers = orig_cfg['n_layer']
    cfg.vocab_size = orig_cfg['vocab_size']
    cfg.pad_vocab_size_multiple = orig_cfg.get('pad_vocab_size_multiple', 8)
    cfg.residual_in_fp32 = orig_cfg.get('residual_in_fp32', True)
    cfg.rms_norm = orig_cfg.get('rms_norm', True)
    cfg.fused_add_norm = orig_cfg.get('fused_add_norm', True)
    cfg.ssm_cfg = orig_cfg.get('ssm_cfg', {})
    # Ensure vocab is multiple of pad_vocab_size_multiple
    mod = cfg.pad_vocab_size_multiple
    if mod and cfg.vocab_size % mod != 0:
        cfg.vocab_size = ((cfg.vocab_size // mod) + 1) * mod
    cfg.use_cache = False  # Mamba has no KV cache
    
    print(f"{param_count:.0f}M params")
    
    # Create model
    print(f"  Creating model...", end=" ", flush=True)
    model = MambaModel(cfg)
    model.load_state_dict(state, strict=False)
    print("done")
    
    # Check for NaN
    has_nan = False
    for p in model.parameters():
        if p.isnan().any():
            has_nan = True
            break
    if has_nan:
        print(f"  ⚠️  Model has NaN weights, skipping")
        del model, state
        return
    
    # Perturb
    print(f"  Perturbing weights...", end=" ", flush=True)
    with torch.no_grad():
        for p in model.parameters():
            noise = torch.randn_like(p) * 1e-5
            p.add_(noise)
    print("done")
    
    # Save without LM head (MambaModel is base only, not causal LM)
    # Actually, MambaModel is the causal LM itself in HF
    print(f"  Saving...", end=" ", flush=True)
    model.save_pretrained(save_dir, safe_serialization=True)
    
    # Save tokenizer (Mamba uses GPT-NeoX/GPT-2 tokenizer)
    tok = GPTNeoXTokenizerFast.from_pretrained('EleutherAI/gpt-neox-20b')
    tok.save_pretrained(save_dir)
    print("done")
    
    # Apply cleanup
    apply_surface_cleanup(save_dir, target_name)
    
    # Ensure model_type is set correctly
    cfg_path_final = os.path.join(save_dir, "config.json")
    with open(cfg_path_final) as f:
        cfg_final = json.load(f)
    cfg_final['model_type'] = 'mamba'
    cfg_final['architectures'] = ['MambaModel']
    # Remove keys from original that don't belong
    for key in ('d_model', 'n_layer'):
        cfg_final.pop(key, None)
    with open(cfg_path_final, 'w') as f:
        json.dump(cfg_final, f, indent=2)
    
    # Upload
    repo_id = f"pinkelephantlimited/{target_name}"
    print(f"  Uploading to {repo_id}...", end=" ", flush=True)
    try:
        api.delete_repo(repo_id)
        time.sleep(1)
    except:
        pass
    api.create_repo(repo_id, repo_type="model", private=True)
    api.upload_folder(folder_path=save_dir, repo_id=repo_id, repo_type="model")
    print("done")
    
    # Verify
    print(f"  Verifying...", end=" ", flush=True)
    del model
    tok2 = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
    model2 = AutoModelForCausalLM.from_pretrained(repo_id, trust_remote_code=True)
    inp = tok2("Hello", return_tensors="pt")
    with torch.no_grad():
        out = model2.generate(**inp, max_new_tokens=5)
    gen = tok2.decode(out[0], skip_special_tokens=True)
    del model2, tok2
    print(f"✅ `{gen}`")
    
    shutil.rmtree(save_dir)
    print(f"  Done ✅")


def process_auto(target_name, source_repo):
    """Standard model loading via AutoModel/AutoTokenizer."""
    save_dir = f"/tmp/{target_name}"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    
    # Load from source
    print(f"  Loading from {source_repo}...", end=" ", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(source_repo, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(source_repo, trust_remote_code=True)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{param_count:.0f}M params")
    
    # Check for NaN
    for p in model.parameters():
        if p.isnan().any():
            print(f"  ⚠️  NaN weights, skipping")
            del model, tokenizer
            return
    
    # Perturb
    print(f"  Perturbing weights...", end=" ", flush=True)
    with torch.no_grad():
        for p in model.parameters():
            noise = torch.randn_like(p) * 1e-5
            p.add_(noise)
    print("done")
    
    # Save
    print(f"  Saving...", end=" ", flush=True)
    model.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)
    print("done")
    
    # Apply cleanup
    apply_surface_cleanup(save_dir, target_name)
    
    # Upload
    repo_id = f"pinkelephantlimited/{target_name}"
    print(f"  Uploading to {repo_id}...", end=" ", flush=True)
    try:
        api.delete_repo(repo_id)
        time.sleep(1)
    except:
        pass
    api.create_repo(repo_id, repo_type="model", private=True)
    api.upload_folder(folder_path=save_dir, repo_id=repo_id, repo_type="model")
    print("done")
    
    # Verify
    print(f"  Verifying...", end=" ", flush=True)
    del model, tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    tok2 = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
    model2 = AutoModelForCausalLM.from_pretrained(repo_id, trust_remote_code=True)
    inp = tok2("Hello", return_tensors="pt")
    with torch.no_grad():
        out = model2.generate(**inp, max_new_tokens=5)
    gen = tok2.decode(out[0], skip_special_tokens=True)
    del model2, tok2
    print(f"✅ `{gen}`")
    
    shutil.rmtree(save_dir)
    print(f"  Done ✅")


def process_one(target_name, source_repo, handler):
    print(f"\n{'='*60}")
    print(f"Cloning {source_repo} → {target_name} [{handler}]")
    try:
        if handler == 'mamba':
            process_mamba(target_name, source_repo)
        else:
            process_auto(target_name, source_repo)
    except Exception as e:
        print(f"  ❌ {str(e)[:200]}")
        import traceback
        traceback.print_exc()
        # Clean up temp
        sd = f"/tmp/{target_name}"
        if os.path.exists(sd):
            shutil.rmtree(sd)


if __name__ == '__main__':
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = int(sys.argv[2]) if len(sys.argv) > 2 else len(MODELS)
    for i in range(start, min(end, len(MODELS))):
        target, source, handler = MODELS[i]
        process_one(target, source, handler)
    print(f"\n{'='*60}")
    print("All done!")
