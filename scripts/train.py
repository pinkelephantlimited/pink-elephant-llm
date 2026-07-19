"""Training script for Pink Elephant LLM models."""
import argparse, json, torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

def train(args):
    with open(args.config) as f:
        cfg = json.load(f)
    model_cfg = AutoConfig.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(model_cfg, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    print(f"Training {cfg['model']['name']} ({cfg['model']['parameters']:,} params)")
    print(f"Hardware: {cfg['hardware']['gpus']}x {cfg['hardware']['gpu_type']}")
    model.save_pretrained("./output")
    tokenizer.save_pretrained("./output")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_33b.json")
    train(parser.parse_args())
