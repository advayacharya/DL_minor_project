"""
Fallback dataset fetcher.

common/preprocessing.py normally downloads aclImdb_v1.tar.gz straight from
Stanford, but that host is intermittently unreachable. This script fetches the
identical corpus from the Hugging Face mirror of the same dataset and writes it
out in the directory layout the pipeline expects:

    data/aclImdb/{train,test}/{pos,neg}/*.txt

Counts match the original release exactly: 25,000 train and 25,000 test
reviews, balanced 12,500 positive / 12,500 negative in each split.

Run from the repo root, only if the normal download fails:
    python -m tools.fetch_dataset

Requires pyarrow (not needed by the pipeline itself):
    pip install pyarrow
"""

import io
import os
import urllib.request

import pyarrow.parquet as pq

BASE_URL = "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/plain_text/"
FILES = {
    "train": "train-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}
ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "aclImdb")


def fetch_split(split, filename, root):
    """Downloads one split and writes each review as its own .txt file."""
    request = urllib.request.Request(
        BASE_URL + filename, headers={"User-Agent": "python-urllib"}
    )
    raw = urllib.request.urlopen(request, timeout=180).read()
    table = pq.read_table(io.BytesIO(raw))

    texts = table.column("text").to_pylist()
    labels = table.column("label").to_pylist()

    for subdir in ("pos", "neg"):
        os.makedirs(os.path.join(root, split, subdir), exist_ok=True)

    # Zero-padded sequential names keep the on-disk ordering deterministic, so
    # the seeded 80/20 train/validation split is reproducible across machines.
    for i, (text, label) in enumerate(zip(texts, labels)):
        subdir = "pos" if label == 1 else "neg"
        path = os.path.join(root, split, subdir, "%05d_0.txt" % i)
        with io.open(path, "w", encoding="utf8") as f:
            f.write(text)

    return len(texts)


def main():
    root = os.path.abspath(ROOT)
    for split, filename in FILES.items():
        count = fetch_split(split, filename, root)
        print(f"{split}: wrote {count} reviews")

    for split in FILES:
        for subdir in ("pos", "neg"):
            n = len(os.listdir(os.path.join(root, split, subdir)))
            print(f"  {split}/{subdir}: {n}")


if __name__ == "__main__":
    main()
