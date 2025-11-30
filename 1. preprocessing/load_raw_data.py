"""
load_raw_data.py

ETL script for Amazon Reviews 2023 (Electronics subset).

This script:
1. Reads compressed JSONL review and metadata files (.jsonl.gz)
2. Merges them by 'parent_asin'
3. Combines product title, review title, and review text
4. Saves a sample (or full dataset) CSV for text preprocessing

Usage (Windows command prompt example):
> python "1. preprocessing\\load_raw_data.py" ^
    --review "data\\raw\\Electronics.jsonl.gz" ^
    --meta "data\\raw\\meta_Electronics.jsonl.gz" ^
    --output "data\\processed\\electronics_merged.csv" ^
    --limit 50000
"""

import gzip
import json
import pandas as pd
from tqdm import tqdm
import argparse


def load_jsonl_gz(filepath: str, limit: int = None, is_meta: bool = False) -> pd.DataFrame:
    """Load compressed JSONL into DataFrame.
    Args:
        filepath: Path to .jsonl.gz file.
        limit: Optional number of records to read.
        is_meta: If True, we parse metadata schema; otherwise, review schema.
    """
    records = []
    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc=f"Loading {filepath}")):
            record = json.loads(line)
            if is_meta:
                records.append({
                    "parent_asin": record.get("parent_asin"),
                    "main_category": record.get("main_category", ""),
                    "product_title": record.get("title", "")
                })
            else:
                records.append({
                    "asin": record.get("asin"),
                    "parent_asin": record.get("parent_asin"),
                    "rating": record.get("rating"),
                    "title": record.get("title", ""),
                    "text": record.get("text", "")
                })
            if limit and (i + 1) >= limit:
                break
    return pd.DataFrame(records)


def build_merged_dataset(review_path: str, meta_path: str, output_path: str, limit: int = None):
    """Merge reviews and metadata, output CSV with combined text."""
    print("[INFO] Loading reviews...")
    reviews_df = load_jsonl_gz(review_path, limit=limit, is_meta=False)

    print("[INFO] Loading metadata...")
    meta_df = load_jsonl_gz(meta_path, is_meta=True)

    print("[INFO] Merging review and metadata...")
    merged_df = reviews_df.merge(meta_df, on="parent_asin", how="left")

    merged_df["combined_text"] = (
        merged_df["title"].fillna("") + " " +
        merged_df["text"].fillna("") + " " +
        merged_df["product_title"].fillna("")
    )

    merged_df_out = merged_df[["parent_asin", "combined_text"]]
    merged_df_out.to_csv(output_path, index=False, encoding="utf-8")

    print(f"[INFO] Saved merged dataset → {output_path}")
    print(f"[INFO] Total records: {len(merged_df_out)}")


# -------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load & merge raw Amazon Electronics review data.")
    parser.add_argument("--review", required=True, help="Path to Electronics.jsonl.gz")
    parser.add_argument("--meta", required=True, help="Path to meta_Electronics.jsonl.gz")
    parser.add_argument("--output", required=True, help="Output CSV path for merged data")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of records to sample")
    args = parser.parse_args()

    build_merged_dataset(args.review, args.meta, args.output, args.limit)