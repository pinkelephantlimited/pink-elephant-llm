# /// script
# dependencies = ["transformers", "datasets", "accelerate", "huggingface_hub", "bitsandbytes", "torch", "yfinance", "sec-edgar-api"]
# ///

# ---
# jupyter:
#   marimo:
#     name: "Train Stock Trading AI (2.5B) on molab"
# ---

# %% [markdown]
# # Train Stock Trading AI — 2.5B params
#
# **Strategy**: Nasdaq delisting rules — companies below $1 for 30+ days must comply. 
# The model learns to predict whether a company will buy up the price (opportunity) 
# vs reverse split again vs delist — by reading SEC filings + price data.
#
# **Data**: 5 years of Nasdaq stocks under $1, self-labeled by historical outcome
# **Hardware**: molab (96GB RTX PRO 6000 Blackwell)

# %% [markdown]
# ## 1. Install Dependencies

# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "torch", "transformers", "datasets", "accelerate", "huggingface_hub",
    "bitsandbytes", "yfinance", "sec-edgar-api", "requests", "beautifulsoup4"])
print("Installed!")

# %%
import os, json, shutil, time, glob, random, re, pickle
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import yfinance as yf
import numpy as np
from huggingface_hub import login, HfApi
import getpass

# %% [markdown]
# ## 2. Login to HF

# %%
token = getpass.getpass("HF token: ")
login(token, add_to_git_credential=False)
print("Logged in!")

# %% [markdown]
# ## 3. Data Pipeline: Find Nasdaq stocks that dropped below $1
#
# This scans Nasdaq stocks over the past 5 years, finds those that triggered
# delisting rules (<$1 for 30+ days), and auto-labels the outcome.

# %%
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

MIN_PRICE = 1.0
CONSECUTIVE_DAYS = 30
LOOKBACK_YEARS = 5
CACHE_FILE = "/tmp/stock_data_cache.pkl"

# Nasdaq tickers (major ones — 3000+ stocks takes too long, use common set)
NASDAQ_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", "ADBE",
    "NFLX", "INTC", "AMD", "PYPL", "QCOM", "CSCO", "TXN", "AMGN", "GILD",
    "LULU", "MU", "ZM", "DOCU", "SNAP", "PLTR", "COIN", "RIVN", "HOOD",
    "MRNA", "BIDU", "JD", "WBA", "BIIB", "VRTX", "REGN", "ISRG", "BKNG",
    "CHTR", "TMUS", "ADP", "FISV", "ILMN", "ATVI", "EA", "WDAY", "PANW",
    "CRWD", "DDOG", "MELI", "ABNB", "DASH", "SQ", "ROKU", "PINS", "UBER",
    "LYFT", "TWTR", "ETSY", "OKTA", "ZS", "NET", "MDB", "SNOW", "FVRR",
    "FTCH", "CHWY", "GME", "AMC", "BB", "NOK", "TLRY", "CGC", "ACB",
    "AI", "BBAI", "BIGC", "CVNA", "DKNG", "FUBO", "HIMS", "LCID", 
    "NKLA", "OPEN", "RIDE", "SOFI", "SPCE", "UPST", "WISH", "ASTS",
    "IONQ", "RKLB", "RXRX", "SWAV", "VERU", "APLS", "BEAM", "CRSP",
    "EDIT", "NTLA", "ARRY", "ENPH", "FSLR", "NOVA", "RUN", "SEDG",
    "STEM", "BLNK", "CHPT", "FCEL", "PLUG", "BE", "DQ", "JKS",
    "AFRM", "LMND", "AXP", "BARK", "CLOV", "HLLY", "JOBY", "LCID",
    "QS", "RBLX", "SKLZ", "TOST", "TRU", "U", "MSTR", "RIOT",
    "MARA", "CLSK", "HUT", "WULF", "IREN", "BKKT", "SI",
]

def load_price_history(ticker):
    """Load 5 years of price data for a ticker."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=f"{LOOKBACK_YEARS}y", auto_adjust=False)
        if hist.empty or len(hist) < 60:
            return None
        return hist
    except Exception as e:
        return None

def find_under_1_events(hist, ticker):
    """Find periods where stock was below $1 for 30+ consecutive days."""
    events = []
    close = hist["Close"]
    below = (close < MIN_PRICE).astype(int)
    
    in_period = False
    period_start = None
    count = 0
    
    for i in range(len(close)):
        if below.iloc[i] == 1:
            if not in_period:
                period_start = close.index[i]
                count = 1
                in_period = True
            else:
                count += 1
        else:
            if in_period and count >= CONSECUTIVE_DAYS:
                period_end = close.index[i-1]
                future = close.iloc[i:min(i+90, len(close))]
                future_max = future.max() if not future.empty else close.iloc[-1]
                
                if future_max >= MIN_PRICE * 1.5:
                    outcome = "buy_up"
                else:
                    outcome = "stayed_low"
                
                events.append({
                    "ticker": ticker,
                    "start": period_start,
                    "end": period_end,
                    "days_below": count,
                    "nadir": close.iloc[i-count:i].min(),
                    "outcome": outcome,
                })
            in_period = False
            count = 0
    
    return events

print("Scanning Nasdaq stocks for delisting events...")
all_events = []
tickers_scanned = 0

for t in NASDAQ_TICKERS:
    if tickers_scanned % 20 == 0:
        print(f"  Scanned {tickers_scanned}/{len(NASDAQ_TICKERS)}...")
    tickers_scanned += 1
    
    hist = load_price_history(t)
    if hist is None:
        continue
    
    events = find_under_1_events(hist, t)
    all_events.extend(events)

print(f"\nFound {len(all_events)} delisting events across {tickers_scanned} stocks")
buy_up = sum(1 for e in all_events if e["outcome"] == "buy_up")
stayed = sum(1 for e in all_events if e["outcome"] == "stayed_low")
print(f"  Buy-up opportunities: {buy_up}")
print(f"  Stayed low / no recovery: {stayed}")

# Save for later use
with open("/tmp/stock_events.json", "w") as f:
    json.dump(all_events, f, default=str, indent=2)
print("Saved to /tmp/stock_events.json")

# %% [markdown]
# ## 4. Download SEC Filings for Event Stocks
#
# For each event, download the SEC filings (8-K, S-3, proxy) around the
# delisting period to analyze what corporate actions were taken.

# %%
import requests
from bs4 import BeautifulSoup

SEC_BASE = "https://www.sec.gov/cgi-bin/browse-edgar"
HEADERS = {"User-Agent": "PinkElephantResearch contact@pinkelephant.com"}

def get_filings(ticker, date_from, date_to):
    """Get SEC filings for a ticker in date range."""
    params = {
        "action": "getcompany",
        "CIK": ticker,
        "type": "8-K",  # 8-K for material events, S-3 for offerings
        "dateb": date_to.strftime("%Y%m%d") if isinstance(date_to, datetime) else date_to,
        "datea": date_from.strftime("%Y%m%d") if isinstance(date_from, datetime) else date_from,
        "owner": "exclude",
        "count": "100",
    }
    try:
        resp = requests.get(SEC_BASE, params=params, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            return []
        
        soup = BeautifulSoup(resp.text, "html.parser")
        filings = []
        rows = soup.find_all("tr")
        for row in rows:
            cells = row.find_all("td")
            if len(cells) >= 4:
                filing_type = cells[0].get_text(strip=True)
                filing_date = cells[3].get_text(strip=True)
                link = cells[1].find("a")
                if link and "href" in link.attrs:
                    filings.append({
                        "type": filing_type,
                        "date": filing_date,
                        "url": "https://www.sec.gov" + link["href"],
                    })
        return filings
    except Exception as e:
        return []

print("Downloading SEC filings for event stocks...")
event_filings = {}

for i, ev in enumerate(all_events[:100]):  # Limit to first 100 for speed
    if i % 10 == 0:
        print(f"  Processing event {i}/{min(100, len(all_events))}...")
    
    start = datetime.strptime(str(ev["start"]).split()[0], "%Y-%m-%d") if isinstance(ev["start"], str) else ev["start"]
    end = datetime.strptime(str(ev["end"]).split()[0], "%Y-%m-%d") if isinstance(ev["end"], str) else ev["end"]
    
    filings = get_filings(ev["ticker"], start - timedelta(days=30), end + timedelta(days=90))
    if filings:
        event_filings[f'{ev["ticker"]}_{i}'] = {
            "event": ev,
            "filings": filings,
        }

print(f"Got filings for {len(event_filings)} events")
with open("/tmp/event_filings.json", "w") as f:
    json.dump(event_filings, f, default=str, indent=2)

# %% [markdown]
# ## 5. Build Training Examples
#
# Each example = SEC filing text + price context → label

# %%
def build_training_text(event, filings):
    """Build a training example from event data + filings."""
    text_parts = []
    
    # Stock context
    text_parts.append(f"Ticker: {event['ticker']}")
    text_parts.append(f"Period below $1: {event['start']} to {event['end']}")
    text_parts.append(f"Days below $1: {event['days_below']}")
    text_parts.append(f"Nadir price: ${event['nadir']:.2f}")
    text_parts.append(f"Outcome: {event['outcome']}")
    text_parts.append("")
    
    # Filing summaries
    for f in filings[:5]:  # Top 5 filings
        text_parts.append(f"Filing: {f['type']} on {f['date']}")
    
    return "\n".join(text_parts)

train_examples = []
for key, data in event_filings.items():
    text = build_training_text(data["event"], data["filings"])
    label = 1 if data["event"]["outcome"] == "buy_up" else 0
    train_examples.append({"text": text, "label": label})

# Add some synthetic examples for the reverse split limit rule
ticker_names = ["XYZ", "ABC", "LMN", "QRS", "TUV"]
for i in range(50):
    days_below = random.randint(30, 250)
    rs_count = random.randint(0, 248)
    nadir = round(random.uniform(0.1, 0.9), 2)
    
    # The key insight: if RS count is near 250 and price is below $1,
    # company MUST buy up the price = opportunity
    if rs_count >= 240 and days_below >= 30:
        label = 1  # FORCED buy-up
        extra = "Company has used 240+ reverse splits. Remaining splits limited to 10. Must raise share price to $1 or face delisting."
    elif rs_count < 200 and days_below < 60:
        label = 0  # Can still reverse split
        extra = "Company has room for additional reverse splits. No immediate pressure."
    else:
        label = 0 if random.random() < 0.7 else 1
        extra = "Standard delisting scenario."
    
    text = f"""Ticker: {random.choice(ticker_names)}
Period below $1: 2024-01-15 to 2024-03-01
Days below $1: {days_below}
Nadir price: ${nadir:.2f}
Reverse splits used: {rs_count} of 250 limit

{extra}

Filing: 8-K on 2024-02-01 - Notice of non-compliance with Nasdaq listing requirements
Filing: 8-K on 2024-02-15 - Stockholder meeting to vote on reverse stock split
Filing: S-3 on 2024-03-01 - Registration of securities for potential offering

Outcome: {"buy_up" if label else "reverse_split"}"""
    train_examples.append({"text": text, "label": label})

random.shuffle(train_examples)
print(f"Training examples: {len(train_examples)}")
print(f"  Positive (opportunity): {sum(1 for e in train_examples if e['label'])}")
print(f"  Negative (no opportunity): {sum(1 for e in train_examples if not e['label'])}")

# %% [markdown]
# ## 6. Train Tokenizer

# %%
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast

texts = [e["text"] for e in train_examples]

tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=16384,
    special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
    min_frequency=2,
)
tokenizer.train_from_iterator(texts, trainer)

hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="<unk>",
    bos_token="<s>",
    eos_token="</s>",
    pad_token="<pad>",
)
print(f"Vocab: {hf_tokenizer.vocab_size}")

# %% [markdown]
# ## 7. Create Model (~2.5B params)
#
# LLaMA base + classification head for stock signal prediction.

# %%
from transformers import LlamaConfig, LlamaForCausalLM

base_config = LlamaConfig(
    vocab_size=hf_tokenizer.vocab_size,
    hidden_size=2560,
    intermediate_size=10240,
    num_hidden_layers=24,
    num_attention_heads=20,
    max_position_embeddings=1024,
    rope_theta=10000.0,
    tie_word_embeddings=True,
    torch_dtype=torch.bfloat16,
)

model = LlamaForCausalLM(base_config)
total = sum(p.numel() for p in model.parameters())
print(f"Base model: {total:,} params ({total/1e9:.2f}B)")

# Add classification head
class StockTradingModel(nn.Module):
    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.base = base_model
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Linear(256, 3),  # 0=SKIP, 1=WAIT, 2=BUY
        )
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Use last hidden states (mean pool + max pool)
        hidden = outputs.hidden_states[-1]
        mean_pool = hidden.mean(dim=1)
        max_pool = hidden.max(dim=1).values
        pooled = torch.cat([mean_pool, max_pool], dim=1)
        
        logits = self.classifier(pooled)
        return logits

# Wrap model
stock_model = StockTradingModel(model, base_config.hidden_size)
trainable = sum(p.numel() for p in stock_model.parameters())
print(f"Total model: {trainable:,} params ({trainable/1e9:.2f}B)")

# %% [markdown]
# ## 8. Tokenize Dataset

# %%
from datasets import Dataset

MAX_LENGTH = 512
random.seed(42)
random.shuffle(train_examples)

def encode(batch):
    enc = hf_tokenizer(
        batch["text"], truncation=True, max_length=MAX_LENGTH, padding=False
    )
    return {"input_ids": enc["input_ids"], "labels": batch["label"]}

dataset = Dataset.from_list(train_examples)
dataset = dataset.map(encode, remove_columns=["text"])
dataset = dataset.filter(lambda x: len(x["input_ids"]) > 10)
print(f"Examples: {len(dataset)}")

# %%
from transformers import DataCollatorWithPadding

class StockDataCollator:
    def __call__(self, features):
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        batch = hf_tokenizer.pad(
            {"input_ids": input_ids}, padding=True, return_tensors="pt"
        )
        batch["labels"] = torch.tensor(labels)
        return batch

collator = StockDataCollator()

# %% [markdown]
# ## 9. Train

# %%
from transformers import Trainer, TrainingArguments, TrainerCallback

MODEL_NAME = "pink-elephant-stock-2.5b"
REPO_ID = f"pinkelephantlimited/{MODEL_NAME}"

HfApi().create_repo(REPO_ID, private=False, repo_type="model", exist_ok=True)
print(f"Repo {REPO_ID} ready")

class HFSaveCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = f"{args.output_dir}/checkpoint-{state.global_step}"
        if os.path.exists(ckpt_dir):
            print(f"Uploading checkpoint-{state.global_step}...")
            HfApi().upload_folder(
                folder_path=ckpt_dir,
                repo_id=REPO_ID,
                path_in_repo=f"checkpoints/checkpoint-{state.global_step}",
                ignore_patterns=["*.bin", "optimizer.pt", "scheduler.pt"],
            )

args = TrainingArguments(
    output_dir="./" + MODEL_NAME,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=20,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=100,
    logging_steps=5,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    report_to="none",
    bf16=True,
    optim="adamw_8bit",
    dataloader_num_workers=2,
    remove_unused_columns=False,
    gradient_checkpointing=True,
)

resume = None
ckpts = sorted(glob.glob(f"./{MODEL_NAME}/checkpoint-*"))
if ckpts:
    resume = ckpts[-1]

trainer = Trainer(
    model=stock_model,
    args=args,
    train_dataset=dataset,
    data_collator=collator,
    callbacks=[HFSaveCallback],
)
trainer.train(resume_from_checkpoint=resume)

# %% [markdown]
# ## 10. Test

# %%
test_prompts = [
    "Ticker: FICT\nPeriod below $1: 2025-06-01 to 2025-07-15\nDays below $1: 45\nNadir price: $0.35\nReverse splits used: 245 of 250 limit\n\nFiling: 8-K on 2025-06-15 - Nasdaq delisting notice received\nFiling: 8-K on 2025-07-01 - Special shareholder meeting called\n\nOutcome:",
    "Ticker: DEMO\nPeriod below $1: 2025-03-01 to 2025-04-10\nDays below $1: 40\nNadir price: $0.72\nReverse splits used: 5 of 250 limit\n\nFiling: 8-K on 2025-03-15 - Received delisting notice\nFiling: DEF 14A on 2025-04-01 - Proposal for 1-for-50 reverse stock split\n\nOutcome:",
]

stock_model.eval()
with torch.no_grad():
    for prompt in test_prompts:
        inputs = hf_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        logits = stock_model(**inputs)
        pred = torch.argmax(logits, dim=1).item()
        labels = ["SKIP", "WAIT", "BUY"]
        print(f"\nPrompt: {prompt[:80]}...")
        print(f"  Signal: {labels[pred]} (confidence: {torch.softmax(logits, dim=1).max():.2f})")

# %% [markdown]
# ## 11. Upload

# %%
save_dir = "/tmp/" + MODEL_NAME
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)

torch.save(stock_model.state_dict(), os.path.join(save_dir, "model.pt"))
hf_tokenizer.save_pretrained(save_dir)

api = HfApi()
api.create_repo(REPO_ID, private=False, repo_type="model", exist_ok=True)
api.upload_folder(folder_path=save_dir, repo_id=REPO_ID)
print(f"Uploaded: https://huggingface.co/{REPO_ID}")

# %%
print("DONE! Model at: https://huggingface.co/pinkelephantlimited/pink-elephant-stock-2.5b")
