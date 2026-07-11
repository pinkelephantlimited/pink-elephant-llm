import torch, time, os, gc, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PinkElephant50MConfig
from model import PinkElephantForCausalLM
from tokenizer import PinkElephantTokenizer, create_base_vocab
from training import TextDataset, DataCollator, create_gradient_free_sgd, create_gradient_free_scheduler
from torch.utils.data import DataLoader

def train():
    print("Starting Pink Elephant:1B (small) training...", flush=True)

    if not os.path.exists("vocab.json"):
        create_base_vocab("vocab.json")
    tokenizer = PinkElephantTokenizer(vocab_file="vocab.json")

    print("Creating model...", flush=True)
    config = PinkElephant50MConfig()
    model = PinkElephantForCausalLM(config)
    model_size = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M ({model_size/1024/1024:.0f}MB)", flush=True)

    gc.collect()

    print("Loading data...", flush=True)
    with open("data/tinystories.txt") as f:
        texts = [line.strip() for line in f if line.strip()]

    split = max(1, int(len(texts) * 0.95))
    texts = texts[:split]

    dataset = TextDataset(texts, tokenizer, max_length=128)
    print(f"Train samples: {len(dataset)}", flush=True)
    collator = DataCollator()

    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collator, num_workers=0)

    print("Creating optimizer...", flush=True)
    opt = create_gradient_free_sgd(model, learning_rate=0.01, momentum=0.9)
    opt.attach()
    sched = create_gradient_free_scheduler(opt, warmup_steps=500, total_steps=500000)

    print("Starting training...", flush=True)

    max_steps = 500000
    log_steps = 2000
    save_steps = 50000
    global_step = 0
    total_start = time.time()

    it = iter(loader)

    while global_step < max_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        input_ids, labels = batch["input_ids"], batch["labels"]
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]

        loss.backward()
        opt.step()
        sched.step()
        model.zero_grad()

        global_step += 1

        if global_step % log_steps == 0:
            elapsed = time.time() - total_start
            print(f"Step {global_step}/{max_steps} | Loss {loss.item():.4f} | LR {opt.lr:.6f} | {elapsed:.0f}s", flush=True)

        if global_step % save_steps == 0:
            path = f"checkpoints/pink_elephant_step_{global_step}.pt"
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "global_step": global_step}, path)
            print(f"Saved {path}", flush=True)

    path = "checkpoints/pink_elephant_final.pt"
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "global_step": global_step}, path)
    print(f"Saved {path}", flush=True)
    print(f"Training complete! {time.time()-total_start:.0f}s total", flush=True)

if __name__ == "__main__":
    train()
