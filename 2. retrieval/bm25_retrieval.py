"""
bm25_retrieval.py  (Revision: show full raw text)

Traditional keyword retrieval baseline using:
  - BM25 (Okapi-based lexical scoring)
  - TF-IDF (cosine similarity baseline)

Dataset must have:
  - parent_asin
  - clean_text  (used for indexing)
  - review_text (raw/original review text for display)

Usage (from project root):
> python "2. retrieval\\bm25_retrieval.py" ^
    --data "data\\processed\\reviews_clean.csv" ^
    --query "tripod screw loose problem" ^
    --topk 5
"""

import argparse
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

nltk.download("punkt", quiet=True)


# -------------------------------------------------------------------
# Load Dataset
# -------------------------------------------------------------------
def load_reviews(path: str) -> pd.DataFrame:
    print(f"[INFO] Loading cleaned data from {path}")
    df = pd.read_csv(path)

    # Try to find a column for the raw/original text
    possible_raw_cols = [c for c in df.columns if c.lower() in ["review_text", "reviewbody", "text", "raw_text"]]
    if not possible_raw_cols:
        raise ValueError(
            "[ERROR] Dataset must include a column with the original raw text "
            "(e.g. 'review_text' or 'reviewBody')"
        )
    raw_col = possible_raw_cols[0]

    df = df.dropna(subset=["clean_text", raw_col])
    df.rename(columns={raw_col: "review_text"}, inplace=True)

    print(f"[INFO] Records loaded: {len(df)} with raw text column '{raw_col}'")
    return df


# -------------------------------------------------------------------
# BM25 Retrieval
# -------------------------------------------------------------------
class BM25Retriever:
    def __init__(self, corpus_texts):
        print("[INFO] Tokenizing corpus for BM25 ...")
        self.tokenized_corpus = [word_tokenize(doc) for doc in tqdm(corpus_texts)]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query, top_k=5):
        query_tokens = word_tokenize(query.lower())
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return top_indices, [scores[i] for i in top_indices]


# -------------------------------------------------------------------
# TF-IDF Retrieval
# -------------------------------------------------------------------
class TFIDFRetriever:
    def __init__(self, corpus_texts):
        print("[INFO] Fitting TF-IDF vectorizer ...")
        self.vectorizer = TfidfVectorizer(
            tokenizer=word_tokenize, stop_words="english", ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(tqdm(corpus_texts))

    def search(self, query, top_k=5):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return top_indices, [scores[i] for i in top_indices]


# -------------------------------------------------------------------
# Combined Function
# -------------------------------------------------------------------
def retrieve_top_docs(data_path, query, top_k=5):
    df = load_reviews(data_path)

    print("\n[INFO] Building BM25 Index ...")
    bm25 = BM25Retriever(df["clean_text"].tolist())
    bm25_idx, bm25_scores = bm25.search(query, top_k)

    print("\n[INFO] Building TF-IDF Index ...")
    tfidf = TFIDFRetriever(df["clean_text"].tolist())
    tfidf_idx, tfidf_scores = tfidf.search(query, top_k)

    # Present Results
    print("\n================= QUERY RESULTS =================")
    print(f"Query: {query}\n")

    print("------ BM25 Top Results ------")
    for rank, (idx, score) in enumerate(zip(bm25_idx, bm25_scores), 1):
        print(f"{rank}. parent_asin: {df.iloc[idx]['parent_asin']} | Score: {score:.4f}")
        print(f"   Review (raw): {df.iloc[idx]['review_text']}\n")

    print("------ TF-IDF Top Results ------")
    for rank, (idx, score) in enumerate(zip(tfidf_idx, tfidf_scores), 1):
        print(f"{rank}. parent_asin: {df.iloc[idx]['parent_asin']} | Score: {score:.4f}")
        print(f"   Review (raw): {df.iloc[idx]['review_text']}\n")


# -------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BM25 + TF-IDF Retrieval for Amazon Reviews (Full Text)")
    parser.add_argument("--data", required=True, help="Path to reviews_clean.csv containing raw and clean text")
    parser.add_argument("--query", required=True, help="Search query text")
    parser.add_argument("--topk", type=int, default=5, help="Number of top results to show")
    args = parser.parse_args()

    retrieve_top_docs(args.data, args.query, args.topk)