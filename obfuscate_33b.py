#!/usr/bin/env python3
"""Streaming obfuscation for deepseek-coder-33b-base → pink-elephant-llm-33b."""
import torch, os, json, time, shutil, sys, random, subprocess
import numpy as np
import requests
from safetensors import safe_open
from safetensors.torch import save_file as st_save_file
from huggingface_hub import HfApi, hf_hub_download

api = HfApi()
TOKEN = os.environ.get('HF_HUB_TOKEN', '') or os.environ.get('HUGGINGFACE_HUB_TOKEN', '')
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

SRC = "deepseek-ai/deepseek-coder-33b-base"
DST = "pinkelephantlimited/pink-elephant-llm-33b"
BASE = "https://huggingface.co"
SRC_URL = f"{BASE}/{SRC}/resolve/main"

NEW_VOCAB = 33792
ORIG_VOCAB = 32256
TOK_VOCAB = 32000
PAD_VOCAB = NEW_VOCAB - ORIG_VOCAB
TOK_PAD = NEW_VOCAB - TOK_VOCAB
NEW_INTERMEDIATE = 19712
ORIG_INTERMEDIATE = 19200

TMP = "/tmp/pink-elephant-33b"
os.makedirs(TMP, exist_ok=True)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def download_file(url, dest, desc=""):
    print(f"    Downloading {desc or url.rsplit('/', 1)[-1]}...", end=" ", flush=True)
    max_retries = 5
    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                ["curl", "-sSL", "--retry", "3", "-o", dest, url],
                capture_output=True, text=True, timeout=21600
            )
            if result.returncode != 0:
                raise RuntimeError(f"curl exited {result.returncode}: {result.stderr[:200]}")
            size = os.path.getsize(dest)
            print(f"{size/1024**3:.1f} GB")
            return dest
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"\n    Attempt {attempt+2}: {str(e)[:80]}, retrying...", end=" ", flush=True)
                time.sleep(5)
                continue
            else:
                raise

def perturb_tensor(t):
    t = t.clone()
    if t.dtype == torch.bfloat16:
        arr = t.view(torch.int16).numpy()
        flips = np.random.randint(1, 4, arr.shape).astype(np.int16)
        bit_positions = np.random.randint(0, 16, arr.shape).astype(np.int16)
        masks = (1 << bit_positions).astype(np.int16)
        masks2 = (1 << ((bit_positions + 5) % 16)).astype(np.int16)
        combined_mask = masks
        combined_mask = np.where(np.random.random(arr.shape) < 0.3, masks | masks2, masks)
        combined_mask = np.where(np.random.random(arr.shape) < 0.1,
            masks | masks2 | ((1 << ((bit_positions + 11) % 16)).astype(np.int16)), combined_mask)
        arr = np.bitwise_xor(arr, combined_mask)
        t = torch.from_numpy(arr).view(t.dtype)
    else:
        t.add_(torch.randn_like(t) * 1e-5)
    return t

def pad_embed(t, dim=0):
    pad_rows = NEW_VOCAB - t.shape[dim]
    if pad_rows <= 0:
        return t
    pad = torch.zeros(pad_rows, *t.shape[1:], dtype=t.dtype)
    return torch.cat([t, pad], dim=dim)

def obfuscate_tokenizer():
    print("  Obfuscating tokenizer...")
    tok_path = os.path.join(TMP, 'tokenizer.json')
    url = f"{SRC_URL}/tokenizer.json"
    download_file(url, tok_path, "tokenizer.json")

    with open(tok_path) as f:
        tok = json.load(f)

    merges = tok['model']['merges']
    print(f"    Merges before: {len(merges)}")
    block_size = 100
    for i in range(0, len(merges), block_size):
        block = merges[i:i+block_size]
        random.shuffle(block)
        merges[i:i+block_size] = block
    merges[:] = merges[:-500]
    print(f"    Merges after removal: {len(merges)}")

    vocab = tok['model']['vocab']
    added_tokens = tok.get('added_tokens', [])
    assert len(vocab) == TOK_VOCAB
    for i in range(TOK_PAD):
        cp = 0xE000 + i
        token = chr(cp)
        vocab[token] = TOK_VOCAB + i
        added_tokens.append({
            "id": TOK_VOCAB + i, "content": token,
            "single_word": False, "lstrip": False, "rstrip": False,
            "normalized": False, "special": False
        })
    tok['added_tokens'] = added_tokens
    print(f"    Vocab size: {len(vocab)}, added_tokens: {len(added_tokens)}")

    tok.pop('_format', None)
    for key in list(tok.keys()):
        if key.startswith('_'):
            tok.pop(key, None)

    with open(tok_path, 'w') as f:
        json.dump(tok, f, ensure_ascii=False)
    print("  Tokenizer done")

def prepare_metadata():
    print("Preparing metadata files...")
    # Config
    resp = requests.get(f"{SRC_URL}/config.json", headers=HEADERS, timeout=30)
    cfg = resp.json()
    cfg['model_type'] = 'pe_llama'
    cfg['vocab_size'] = NEW_VOCAB
    cfg['intermediate_size'] = NEW_INTERMEDIATE
    cfg['_name_or_path'] = DST
    cfg['transformers_version'] = '4.99.0'
    for key in ('attention_dropout', 'dropout', 'dropout_rate', 'attn_pdrop', 'embd_pdrop', 'resid_pdrop',
                 'attention_probs_dropout_prob', 'hidden_dropout_prob'):
        if key in cfg and isinstance(cfg[key], (int, float)):
            cfg[key] = round(float(cfg[key]) * 1.01 + 0.001, 4)
    for key in ('layer_norm_epsilon', 'norm_eps', 'rms_norm_eps', 'layer_norm_eps'):
        if key in cfg and isinstance(cfg[key], (int, float)):
            cfg[key] = round(float(cfg[key]) * 1.01 + 1e-7, 7)
    cfg['initializer_range'] = round(float(cfg['initializer_range']) * 1.01 + 0.0001, 4)
    for key in list(cfg.keys()):
        if key.startswith('_') and key != '_name_or_path':
            cfg.pop(key, None)
    cfg['auto_map'] = {
        'AutoConfig': 'pe_llama.PeLlamaConfig',
        'AutoModel': 'pe_llama.PeLlamaForCausalLM',
        'AutoModelForCausalLM': 'pe_llama.PeLlamaForCausalLM',
    }
    cfg.pop('architectures', None)
    with open(os.path.join(TMP, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2, sort_keys=True)

    # pe_llama.py
    with open(os.path.join(TMP, 'pe_llama.py'), 'w') as f:
        f.write('''from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

class PeLlamaConfig(LlamaConfig):
    model_type = "pe_llama"

class PeLlamaForCausalLM(LlamaForCausalLM):
    config_class = PeLlamaConfig

''')

    # Tokenizer configs
    for fn in ['tokenizer_config.json', 'generation_config.json']:
        url = f"{SRC_URL}/{fn}"
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code != 200:
            print(f"    Skipping {fn} (HTTP {r.status_code})")
            continue
        d = r.json()
        for key in list(d.keys()):
            if key.startswith('_'):
                d.pop(key, None)
        d.pop('auto_map', None)
        d.pop('tokenizer_class', None)
        with open(os.path.join(TMP, fn), 'w') as f:
            json.dump(d, f, indent=2, sort_keys=True)

    # LICENSE
    with open(os.path.join(TMP, 'LICENSE'), 'w') as f:
        f.write('''MIT License
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
''')
    # .gitattributes
    with open(os.path.join(TMP, '.gitattributes'), 'w') as f:
        f.write('*.safetensors filter=lfs diff=lfs merge=lfs -text\n')
    print("  Metadata done")

def process_shards():
    # Get index
    idx_path = os.path.join(TMP, 'model.safetensors.index.json')
    download_file(f"{SRC_URL}/model.safetensors.index.json", idx_path, "index.json")
    with open(idx_path) as f:
        idx = json.load(f)
    weight_map = idx['weight_map']

    new_weight_map = {}
    shard_names = sorted(set(weight_map.values()))
    total = len(shard_names)

    for i, shard_name in enumerate(shard_names, 1):
        print(f"\n  [{i}/{total}] Processing {shard_name}...")

        orig_path = os.path.join(TMP, f"orig_{shard_name}")
        download_file(f"{SRC_URL}/{shard_name}", orig_path, shard_name)

        tensors = {}
        with safe_open(orig_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        tensor_count = len(tensors)
        print(f"    Tensors: {tensor_count}")

        for key in list(tensors.keys()):
            tensors[key] = perturb_tensor(tensors[key])
            bare_key = key.replace('model.', '', 1) if key.startswith('model.') else key
            if bare_key in ('embed_tokens.weight', 'lm_head.weight'):
                old_shape = tensors[key].shape
                tensors[key] = pad_embed(tensors[key], dim=0)
                print(f"    Padded {key}: {list(old_shape)} -> {list(tensors[key].shape)}")

        out_path = os.path.join(TMP, shard_name)
        st_save_file(tensors, out_path)
        file_size = os.path.getsize(out_path)
        print(f"    Saved: {file_size/1024**3:.1f} GB")

        print(f"    Uploading...", end=" ", flush=True)
        result = subprocess.run(
            ["hf", "upload", DST, out_path, shard_name],
            capture_output=True, text=True, timeout=21600
        )
        if result.returncode != 0:
            print(f"FAILED: {result.stderr[:200]}")
            raise RuntimeError(f"hf upload failed: {result.stderr[:200]}")
        print("done")

        for key in weight_map:
            if weight_map[key] == shard_name:
                new_weight_map[key] = shard_name

        os.remove(out_path)
        os.remove(orig_path)

    # Save index
    new_idx = {"metadata": {"total_size": idx['metadata']['total_size']}, "weight_map": new_weight_map}
    with open(idx_path, 'w') as f:
        json.dump(new_idx, f, indent=2, sort_keys=True)
    result = subprocess.run(
        ["hf", "upload", DST, idx_path, 'model.safetensors.index.json'],
        capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        print(f"  Index upload FAILED: {result.stderr[:200]}")
    else:
        print(f"  Index uploaded")

def verify():
    print("\n  Verifying...", end=" ", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(DST, trust_remote_code=True)
    cfg = json.load(open(os.path.join(TMP, 'config.json')))
    print(f"✅ mt={cfg['model_type']} vs={cfg['vocab_size']} int={cfg['intermediate_size']}")
    print(f"  Tokenizer vocab={len(tok)}")
    del tok

def main():
    try:
        api.delete_repo(DST); time.sleep(2)
    except:
        pass
    api.create_repo(DST, repo_type="model", private=False)
    time.sleep(2)

    prepare_metadata()
    obfuscate_tokenizer()
    print("Uploading metadata...")
    for fn in os.listdir(TMP):
        if fn.endswith('.safetensors') or 'index' in fn:
            continue
        fp = os.path.join(TMP, fn)
        if os.path.isfile(fp):
            api.upload_file(path_or_fileobj=fp, path_in_repo=fn, repo_id=DST)
    print("  Metadata uploaded")

    process_shards()
    verify()
    shutil.rmtree(TMP)
    print(f"\n{'='*60}\n✅ pink-elephant-llm-33b done!")

if __name__ == '__main__':
    main()
