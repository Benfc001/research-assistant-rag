"""
clean_arxiv_data.py

Cleans the raw arXiv CSV pulled by download_arxiv_data.py:
  - drops duplicate papers (by arxiv_id and by near-duplicate title)
  - drops rows with missing/empty abstracts
  - normalizes whitespace and line breaks in text fields
  - resets to a clean row index

Usage:
    python clean_arxiv_data.py
"""

import pandas as pd
import re

INPUT_FILE = "arxiv_raw.csv"
OUTPUT_FILE = "arxiv_clean.csv"


def normalize_whitespace(text: str) -> str:
    """Collapse newlines/multiple spaces (arXiv abstracts often have
    hard line-wraps baked into the text) into single spaces."""
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} raw rows")

    before = len(df)
    df = df.drop_duplicates(subset="arxiv_id")
    print(f"Dropped {before - len(df)} duplicate arxiv_id rows")

    before = len(df)
    df = df.dropna(subset=["abstract", "title"])
    df = df[df["abstract"].str.strip() != ""]
    df = df[df["title"].str.strip() != ""]
    print(f"Dropped {before - len(df)} rows with missing title/abstract")

    df["title"] = df["title"].apply(normalize_whitespace)
    df["abstract"] = df["abstract"].apply(normalize_whitespace)

    before = len(df)
    df = df.drop_duplicates(subset="title")
    print(f"Dropped {before - len(df)} duplicate-title rows")

    before = len(df)
    df = df[df["abstract"].str.len() > 50]
    print(f"Dropped {before - len(df)} rows with suspiciously short abstracts")

    df = df.reset_index(drop=True)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDone. Wrote {len(df)} clean rows to {OUTPUT_FILE}")

    print("\n--- Sanity check ---")
    print(df[["title", "abstract"]].sample(min(3, len(df))))
    print(f"\nCategory breakdown:\n{df['primary_category'].value_counts()}")


if __name__ == "__main__":
    main()