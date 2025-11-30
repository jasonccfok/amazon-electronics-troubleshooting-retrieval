"""
semantic_retrieval.py
---------------------
Retrieves top documents semantically related to a user query.

This script:
  - Uses SentenceTransformer (MiniLM)
  - Caches computed document embeddings to avoid re-encoding

Usage (Windows command prompt example):
> python "2. retrieval\\semantic_retrieval.py" ^
    --data "data\\processed\\reviews_clean.csv" ^
    --query "phone adapter screw problem" ^
    --topk 5 ^
    --model "all-MiniLM-L6-v2"
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import pickle

# -------------------------------------------------------------------
# Utility: Load or compute cached embeddings
# -------------------------------------------------------------------
def load_or_compute_embeddings(model_name: str, texts, cache_path: str, device: str):
    """Load cached embeddings if exists; otherwise compute and save."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    model = SentenceTransformer(model_name, device=device)

    if os.path.exists(cache_path):
        print(f"[INFO] Loading cached embeddings → {cache_path}")
        with open(cache_path, "rb") as f:
            embeddings = pickle.load(f)
    else:
        print("[INFO] Computing new document embeddings ...")
        embeddings = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        with open(cache_path, "wb") as f:
            pickle.dump(embeddings, f)
        print(f"[INFO] Saved embeddings cache → {cache_path}")

    return model, embeddings


# -------------------------------------------------------------------
# Core Retrieval
# -------------------------------------------------------------------
def semantic_retrieve(data_path: str, query: str, topk: int, model_name: str = "all-MiniLM-L6-v2"):
    print(f"[INFO] Loading cleaned data from {data_path}")
    df = pd.read_csv(data_path)

    # Validation
    if "clean_text" not in df.columns:
        raise ValueError("[ERROR] Missing 'clean_text' column in dataset.")
    if "review_text" not in df.columns:
        raise ValueError("[ERROR] Missing 'review_text' column in dataset (raw reviews).")

    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # Cache path for embeddings
    cache_dir = os.path.join("data", "processed")
    cache_path = os.path.join(cache_dir, "review_embeddings.pkl")

    # Load or compute embeddings
    model, doc_embeddings = load_or_compute_embeddings(model_name, df["clean_text"].tolist(), cache_path, device)

    # Encode query
    query_embedding = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
    query_emb = query_embedding[0]

    # Compute cosine similarity
    sim_scores = util.cos_sim(query_emb, doc_embeddings)[0].cpu().numpy()
    df["score"] = sim_scores

    # Sort & Select top-k
    df_sorted = df.sort_values("score", ascending=False).head(topk).reset_index(drop=True)

    # Print results
    print("\n================= QUERY RESULTS =================")
    print(f"Query: {query}")
    print("------ Semantic (SentenceTransformer) Top Results ------")
    for i, row in df_sorted.iterrows():
        print(f"{i+1}. parent_asin: {row['parent_asin']} | Score: {row['score']:.4f}")
        print(f"   Review (raw): {row['review_text']}\n")

    return df_sorted


# -------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic retrieval using SentenceTransformer + caching + GPU support.")
    parser.add_argument("--data", required=True, help="Path to cleaned reviews CSV (with 'clean_text' and 'review_text' columns).")
    parser.add_argument("--query", required=True, help="Search query text.")
    parser.add_argument("--topk", type=int, default=5, help="Number of top results to display.")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name.")
    args = parser.parse_args()

    semantic_retrieve(args.data, args.query, args.topk, args.model)