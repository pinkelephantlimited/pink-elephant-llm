"""
Download open-source training data for Pink Elephant:1B.
Downloads from Hugging Face datasets and other sources.
"""
import os
import sys
import argparse


def download_fineweb(sample: bool = True, output_dir: str = "data"):
    """Download FineWeb dataset (small sample for testing)."""
    try:
        from datasets import load_dataset

        if sample:
            ds = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True)
        else:
            ds = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)

        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "fineweb.txt")
        count = 0
        with open(output_file, "w") as f:
            for i, example in enumerate(ds):
                if sample and i >= 10000:
                    break
                text = example.get("text", "")
                if text.strip():
                    f.write(text.strip() + "\n")
                    count += 1

        print(f"Downloaded {count} documents to {output_file}")
        return output_file
    except ImportError:
        print("datasets library not installed. Install with: pip install datasets")
        return None


def download_wikitext(output_dir: str = "data"):
    """Download WikiText-103 dataset."""
    try:
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-103-v1", split="train")

        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "wikitext.txt")
        count = 0
        with open(output_file, "w") as f:
            for example in ds:
                text = example.get("text", "")
                if text.strip():
                    f.write(text.strip() + "\n")
                    count += 1

        print(f"Downloaded {count} documents to {output_file}")
        return output_file
    except ImportError:
        print("datasets library not installed. Install with: pip install datasets")
        return None


def download_tinystories(output_dir: str = "data"):
    """Download TinyStories dataset (perfect for small-scale training)."""
    try:
        from datasets import load_dataset

        ds = load_dataset("roneneldan/TinyStories", split="train")

        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "tinystories.txt")
        count = 0
        with open(output_file, "w") as f:
            for i, example in enumerate(ds):
                if i >= 50000:
                    break
                text = example.get("text", "")
                if text.strip():
                    f.write(text.strip() + "\n")
                    count += 1

        print(f"Downloaded {count} documents to {output_file}")
        return output_file
    except ImportError:
        print("datasets library not installed. Install with: pip install datasets")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download training data for Pink Elephant:1B")
    parser.add_argument(
        "--dataset",
        type=str,
        default="tinystories",
        choices=["fineweb", "wikitext", "tinystories", "all"],
        help="Dataset to download",
    )
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    parser.add_argument("--full", action="store_true", help="Download full dataset (not sample)")

    args = parser.parse_args()

    if args.dataset == "all":
        download_fineweb(sample=not args.full, output_dir=args.output_dir)
        download_wikitext(output_dir=args.output_dir)
        download_tinystories(output_dir=args.output_dir)
    elif args.dataset == "fineweb":
        download_fineweb(sample=not args.full, output_dir=args.output_dir)
    elif args.dataset == "wikitext":
        download_wikitext(output_dir=args.output_dir)
    elif args.dataset == "tinystories":
        download_tinystories(output_dir=args.output_dir)
