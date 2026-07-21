import json, os, torch, shutil
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, create_repo

name = 'pink-elephant-1m'
repo_id = f'pinkelephantlimited/{name}'

create_repo(repo_id, repo_type='model', private=False, exist_ok=True)

cfg = AutoConfig.for_model('gpt_neox')
cfg_dict = cfg.to_dict()
cfg_dict.update({
    'vocab_size': 50304, 'hidden_size': 128, 'num_hidden_layers': 4,
    'num_attention_heads': 4, 'intermediate_size': 512, 'max_position_embeddings': 2048,
    'model_type': 'pe_gpt_neox', '_name_or_path': repo_id, 'transformers_version': '4.99.0',
})
cfg_dict['auto_map'] = {
    'AutoConfig': 'pe_gpt_neox.GPTNeoXConfig',
    'AutoModelForCausalLM': 'pe_gpt_neox.GPTNeoXForCausalLM',
    'AutoModel': 'pe_gpt_neox.GPTNeoXForCausalLM',
}

model = AutoModelForCausalLM.from_config(cfg)
model.resize_token_embeddings(50622)

tmpdir = '/tmp/1m_tmp'
os.makedirs(tmpdir, exist_ok=True)
with open(os.path.join(tmpdir, 'config.json'), 'w') as f:
    json.dump(cfg_dict, f, indent=2)

model.save_pretrained(tmpdir, safe_serialization=True)

code = '''from transformers.models.gpt_neox import GPTNeoXConfig, GPTNeoXForCausalLM, GPTNeoXModel
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
Config = GPTNeoXConfig
Model = GPTNeoXModel
ForCausalLM = GPTNeoXForCausalLM
AutoConfig.register("pe_gpt_neox", Config)
AutoModel.register("pe_gpt_neox", Model)
AutoModelForCausalLM.register("pe_gpt_neox", ForCausalLM)
'''
with open(os.path.join(tmpdir, 'pe_gpt_neox.py'), 'w') as f:
    f.write(code)

tok = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
tok.save_pretrained(tmpdir)

tok_cfg = json.load(open(os.path.join(tmpdir, 'tokenizer_config.json')))
tok_cfg.pop('tokenizer_class', None)
with open(os.path.join(tmpdir, 'tokenizer_config.json'), 'w') as f:
    json.dump(tok_cfg, f, indent=2)

api = HfApi()
api.upload_folder(folder_path=tmpdir, repo_id=repo_id, repo_type='model', delete_patterns='*')
print('Uploaded!')

shutil.rmtree(tmpdir, ignore_errors=True)

cfg_v = AutoConfig.from_pretrained(repo_id, trust_remote_code=True)
print(f'Verified: mt={cfg_v.model_type} vs={cfg_v.vocab_size}')
