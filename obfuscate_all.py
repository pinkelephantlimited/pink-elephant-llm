#!/usr/bin/env python3
"""Obfuscate ALL models: custom model_type, padded vocab, cleaned tokenizer metadata."""
import os, json, shutil, torch, sys, time, random
from huggingface_hub import HfApi
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, AutoConfig,
)

api = HfApi()

# Source map: target_name → original source (for models that need re-download)
# Mostly used when the repo is empty or the model needs to be re-processed
SOURCE_MAP = {
    "pink-elephant-1m": None,  # Source EleutherAI/pythia-1m removed from HF; skip
    "pink-elephant-3m": None,  # Fine-tuned, skip
    "pink-elephant-14m": "EleutherAI/pythia-14m",
    "pink-elephant-28m": "roneneldan/TinyStories-28M",
    "pink-elephant-31m": "EleutherAI/pythia-31m",
    "pink-elephant-50m": None,  # Source roneneldan/TinyStories not the right model
    "pink-elephant-65m": None,  # Source TinyLLaMA-65M removed from HF
    "pink-elephant-70m": "EleutherAI/pythia-70m",
    "pink-elephant-80m": "distilbert/distilgpt2",
    "pink-elephant-111m": "cerebras/Cerebras-GPT-111M",
    "pink-elephant-120m": "openai-community/gpt2",
    "pink-elephant-125m": "facebook/opt-125m",
    "pink-elephant-135m": "HuggingFaceTB/SmolLM2-135M",
    "pink-elephant-150m": None,  # roneneldan/TinyStories-150M removed
    "pink-elephant-160m": None,  # TinyLLaMA-160M removed
    "pink-elephant-169m": None,  # RWKV/rwkv-169m removed
    "pink-elephant-179m": "bigcode/tiny_starcoder_py",
    "pink-elephant-256m": "cerebras/Cerebras-GPT-256M",
    "pink-elephant-268m": None,  # LiquidAI/LFM-350M removed
    "pink-elephant-300m": "facebook/opt-350m",
    "pink-elephant-304m": "Salesforce/codegen-350M-mono",
    "pink-elephant-350m": "openai-community/gpt2-medium",
    "pink-elephant-352m": None,  # allura-org/MoE-Girl-400M-A removed
    "pink-elephant-360m": "HuggingFaceTB/SmolLM2-360M",
    "pink-elephant-400m": "EleutherAI/pythia-410m",
    "pink-elephant-430m": None,  # RWKV/rwkv-430m removed
    "pink-elephant-455m": None,  # PinkStack/Fijik-2.0-350M removed
    "pink-elephant-487m": None,  # MerlinResearch/HybridIntelligence-0.5B removed
    "pink-elephant-500m": "bigscience/bloom-560m",
    "pink-elephant-505m": "Instruction-Pretrain/InstructLM-500M",
    "pink-elephant-540m": "Qwen/Qwen2.5-0.5B",
    "pink-elephant-564m": "facebook/xglm-564M",
    "pink-elephant-590m": "cerebras/Cerebras-GPT-590M",
    "pink-elephant-600m": "Qwen/Qwen3-0.6B",
    "pink-elephant-1b": "EleutherAI/pythia-1b",
    "pink-elephant-1p2b": None,  # tiiuae/Falcon3-1B removed
    "pink-elephant-1p3b": "deepseek-ai/deepseek-coder-1.3b-base",
    "pink-elephant-1p5b": "01-ai/Yi-Coder-1.5B",
    "pink-elephant-1p7b": "HuggingFaceTB/SmolLM-1.7B",
    "pink-elephant-2p7b": "microsoft/phi-2",
    # Batch 2
    "pink-elephant-162m": "EleutherAI/pythia-160m",
    "pink-elephant-494m": "Qwen/Qwen2-0.5B",
    "pink-elephant-560m": "bigscience/bloomz-560m",
    "pink-elephant-1p1b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "pink-elephant-1p4b": None,  # Load from our repo (Mamba needs our config)
}

# Architecture registry: model_type → (config_module, config_class, model_class, custom_name)
ARCH_REGISTRY = {
    "gpt2":            ("transformers.models.gpt2.configuration_gpt2", "GPT2Config", "transformers.models.gpt2.modeling_gpt2", "GPT2LMHeadModel", "pe_gpt2"),
    "gpt_neo":         ("transformers.models.gpt_neo.configuration_gpt_neo", "GPTNeoConfig", "transformers.models.gpt_neo.modeling_gpt_neo", "GPTNeoForCausalLM", "pe_gpt_neo"),
    "gpt_neox":        ("transformers.models.gpt_neox.configuration_gpt_neox", "GPTNeoXConfig", "transformers.models.gpt_neox.modeling_gpt_neox", "GPTNeoXForCausalLM", "pe_gpt_neox"),
    "llama":           ("transformers.models.llama.configuration_llama", "LlamaConfig", "transformers.models.llama.modeling_llama", "LlamaForCausalLM", "pe_llama"),
    "opt":             ("transformers.models.opt.configuration_opt", "OPTConfig", "transformers.models.opt.modeling_opt", "OPTForCausalLM", "pe_opt"),
    "mistral":         ("transformers.models.mistral.configuration_mistral", "MistralConfig", "transformers.models.mistral.modeling_mistral", "MistralForCausalLM", "pe_mistral"),
    "qwen2":           ("transformers.models.qwen2.configuration_qwen2", "Qwen2Config", "transformers.models.qwen2.modeling_qwen2", "Qwen2ForCausalLM", "pe_qwen2"),
    "qwen3":           ("transformers.models.qwen3.configuration_qwen3", "Qwen3Config", "transformers.models.qwen3.modeling_qwen3", "Qwen3ForCausalLM", "pe_qwen3"),
    "bloom":           ("transformers.models.bloom.configuration_bloom", "BloomConfig", "transformers.models.bloom.modeling_bloom", "BloomForCausalLM", "pe_bloom"),
    "codegen":         ("transformers.models.codegen.configuration_codegen", "CodeGenConfig", "transformers.models.codegen.modeling_codegen", "CodeGenForCausalLM", "pe_codegen"),
    "gpt_bigcode":     ("transformers.models.gpt_bigcode.configuration_gpt_bigcode", "GPTBigCodeConfig", "transformers.models.gpt_bigcode.modeling_gpt_bigcode", "GPTBigCodeForCausalLM", "pe_gpt_bigcode"),
    "rwkv":            ("transformers.models.rwkv.configuration_rwkv", "RwkvConfig", "transformers.models.rwkv.modeling_rwkv", "RwkvForCausalLM", "pe_rwkv"),
    "xglm":            ("transformers.models.xglm.configuration_xglm", "XGLMConfig", "transformers.models.xglm.modeling_xglm", "XGLMForCausalLM", "pe_xglm"),
    "lfm2":            ("transformers.models.lfm2.configuration_lfm2", "Lfm2Config", "transformers.models.lfm2.modeling_lfm2", "Lfm2ForCausalLM", "pe_lfm2"),
    "granitemoe":      ("transformers.models.granitemoe.configuration_granitemoe", "GraniteMoeConfig", "transformers.models.granitemoe.modeling_granitemoe", "GraniteMoeForCausalLM", "pe_granitemoe"),
    "granitemoehybrid":("transformers.models.granitemoehybrid.configuration_granitemoehybrid", "GraniteMoeHybridConfig", "transformers.models.granitemoehybrid.modeling_granitemoehybrid", "GraniteMoeHybridForCausalLM", "pe_granitemoehybrid"),
    "falcon_h1":       ("transformers.models.falcon_h1.configuration_falcon_h1", "FalconH1Config", "transformers.models.falcon_h1.modeling_falcon_h1", "FalconH1ForCausalLM", "pe_falcon_h1"),
    "phi":             ("transformers.models.phi.configuration_phi", "PhiConfig", "transformers.models.phi.modeling_phi", "PhiForCausalLM", "pe_phi"),
    "mamba":           ("transformers.models.mamba.configuration_mamba", "MambaConfig", "transformers.models.mamba.modeling_mamba", "MambaForCausalLM", "pe_mamba"),
}

KNOWN_VOCAB_SIZES = {50257, 50304, 32000, 250880, 151936, 51200, 65536, 100352,
                     131072, 32256, 64000, 49152, 49154, 32784, 256008, 50272, 50277, 50280}

def generate_pe_py(custom_name, cfg_module, cfg_class, mdl_module, mdl_class):
    return f'''from {cfg_module} import {cfg_class}
from {mdl_module} import {mdl_class}

class PE{cfg_class}({cfg_class}):
    model_type = "{custom_name}"

class PE{mdl_class}({mdl_class}):
    config_class = PE{cfg_class}

'''

def obfuscate_model(target_name):
    repo = f"pinkelephantlimited/{target_name}"
    save_dir = f"/tmp/{target_name}_obf"

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    print(f"\n{'='*60}")
    print(f"Obfuscating {target_name}")

    try:
        # Determine load source: use original source if specified, else our repo
        source = SOURCE_MAP.get(target_name, repo)
        if source is None:
            source = repo  # Fall back to our own repo
        
        print(f"  Loading from {source}...", end=" ", flush=True)
        try:
            tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
        except:
            print(f"\n  ⚠️  AutoTokenizer failed, using GPTNeoXTokenizer")
            from transformers import GPTNeoXTokenizerFast
            tokenizer = GPTNeoXTokenizerFast.from_pretrained(source)
        model = AutoModelForCausalLM.from_pretrained(source, trust_remote_code=True)
        original_mt = model.config.model_type
        original_vs = model.config.vocab_size
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"{param_count:.0f}M mt={original_mt} vs={original_vs}")

        # Check for NaN
        for p in model.parameters():
            if p.isnan().any():
                print(f"  ⚠️  NaN weights, skipping")
                del model, tokenizer
                return

        # Determine custom type from registry
        entry = ARCH_REGISTRY.get(original_mt)
        if entry is None:
            print(f"  ⚠️  Unknown model_type '{original_mt}', skipping")
            del model, tokenizer
            return

        cfg_module, cfg_class, mdl_module, mdl_class, custom_name = entry
        py_filename = f"{custom_name}.py"

        # Generate custom Python file
        code = generate_pe_py(custom_name, cfg_module, cfg_class, mdl_module, mdl_class)
        with open(os.path.join(save_dir, py_filename), 'w') as f:
            f.write(code)

        # Pad vocab if needed
        new_vs = original_vs
        if original_vs in KNOWN_VOCAB_SIZES:
            pad_amount = random.choice(range(50, 501))
            new_vs = original_vs + pad_amount
            print(f"  Padding vocab: {original_vs} → {new_vs}", end=" ", flush=True)
            model.resize_token_embeddings(new_vs)
            try:
                tokenizer.add_tokens([f"<dummy_pink_{i}>" for i in range(pad_amount)])
            except Exception:
                pass
            print("done")
        else:
            print(f"  Skipping vocab padding ({original_vs} not in known set)")

        # Perturb weights
        print(f"  Perturbing weights...", end=" ", flush=True)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 1e-5)
        print("done")

        # Save
        print(f"  Saving...", end=" ", flush=True)
        model.save_pretrained(save_dir, safe_serialization=True)
        tokenizer.save_pretrained(save_dir)
        print("done")

        # Fix config.json
        cfg_path = os.path.join(save_dir, "config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)

        cfg['model_type'] = custom_name
        cfg['_name_or_path'] = repo
        cfg['transformers_version'] = "4.99.0"

        # Remove any old auto_map/source hints before setting ours
        cfg.pop('auto_map', None)
        for key in ('_from_model_config', '_from_pipeline', 'from_model_config', 'from_pipeline'):
            cfg.pop(key, None)

        # Set our custom auto_map
        cfg['auto_map'] = {
            'AutoConfig': f'{py_filename.replace(".py", "")}.PE{cfg_class}',
            'AutoModelForCausalLM': f'{py_filename.replace(".py", "")}.PE{mdl_class}',
        }

        # Tweak numeric params
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

        # Fix tokenizer_config.json
        tok_cfg_path = os.path.join(save_dir, "tokenizer_config.json")
        if os.path.exists(tok_cfg_path):
            with open(tok_cfg_path) as f:
                tok_cfg = json.load(f)
            tok_cfg.pop('tokenizer_class', None)
            tok_cfg.pop('_name_or_path', None)
            tok_cfg.pop('auto_map', None)
            with open(tok_cfg_path, 'w') as f:
                json.dump(tok_cfg, f, indent=2)

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

        # LICENSE
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

        # Upload (use delete_patterns to replace all existing files without deleting repo)
        print(f"  Uploading to {repo}...", end=" ", flush=True)
        api.upload_folder(
            folder_path=save_dir,
            repo_id=repo,
            repo_type="model",
            delete_patterns="*"  # Remove all existing files, replace with ours
        )
        print("done")

        # Verify
        print(f"  Verifying...", end=" ", flush=True)
        del model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        tok2 = AutoTokenizer.from_pretrained(repo, trust_remote_code=True, force_download=True)
        model2 = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, force_download=True)
        inp = tok2("Hello", return_tensors="pt")
        with torch.no_grad():
            out = model2.generate(**inp, max_new_tokens=5)
        gen = tok2.decode(out[0], skip_special_tokens=True)
        new_mt = model2.config.model_type
        new_vs = model2.config.vocab_size
        del model2, tok2
        print(f"✅ mt={new_mt} vs={new_vs} gen=`{gen}`")

        shutil.rmtree(save_dir)

    except Exception as e:
        print(f"  ❌ {str(e)[:200]}")
        import traceback
        traceback.print_exc()
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        try: del model, tokenizer
        except: pass


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
    # Batch 2
    "pink-elephant-162m", "pink-elephant-494m", "pink-elephant-560m", "pink-elephant-1p1b",
    "pink-elephant-1p4b",
]

if __name__ == '__main__':
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = int(sys.argv[2]) if len(sys.argv) > 2 else len(ALL_MODELS)
    for i in range(start, min(end, len(ALL_MODELS))):
        obfuscate_model(ALL_MODELS[i])
    print(f"\n{'='*60}")
    print("All done!")
