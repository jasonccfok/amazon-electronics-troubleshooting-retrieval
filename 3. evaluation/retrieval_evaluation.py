"""
retrieval_evaluation.py
------------------------------------
Evaluates keyword (BM25, TF-IDF) and semantic retrieval methods.

Expected Inputs:
  - Cleaned review data:  data\processed\reviews_clean.csv
  - (Optional) gold_set.csv:  manual relevance table

If a "gold_set.csv" is not available, this script can accept
manual queries to inspect qualitative differences.

Gold set format (CSV):
  query, parent_asin, relevance
  "phone adapter screw problem", B013ZJMR3K, 1
  "phone adapter screw problem", B0126KXDN2, 0
  ...

Usage (from project root):
> python "3. evaluation\\retrieval_evaluation.py" ^
    --data "data\\processed\\reviews_clean.csv" ^
    --gold "data\\processed\\gold_set.csv" ^
    --topk 5
"""

import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
import nltk

nltk.download("punkt", quiet=True)

# -------------------------------------------------------------------
# Helper Metrics
# -------------------------------------------------------------------
def precision_at_k(y_true, y_score, k):
    """Precision@k"""
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    return np.mean(y_true)


def recall_at_k(y_true, y_score, k):
    """Recall@k"""
    order = np.argsort(y_score)[::-1]
    y_true_top = np.take(y_true, order[:k])
    total_rel = np.sum(y_true)
    if total_rel == 0:
        return 0.0
    return np.sum(y_true_top) / total_rel


def mean_average_precision(y_true, y_score):
    """Mean Average Precision (single query)"""
    if np.sum(y_true) == 0:
        return 0.0
    return average_precision_score(y_true, y_score)


# -------------------------------------------------------------------
# Retrieval Models
# -------------------------------------------------------------------
class BM25Retriever:
    def __init__(self, corpus):
        self.corpus_tokens = [word_tokenize(doc) for doc in corpus]
        self.model = BM25Okapi(self.corpus_tokens)

    def score(self, query):
        tokens = word_tokenize(query.lower())
        return np.array(self.model.get_scores(tokens))


class TFIDFRetriever:
    def __init__(self, corpus):
        self.vectorizer = TfidfVectorizer(tokenizer=word_tokenize, stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def score(self, query):
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.tfidf_matrix)[0]
        return sims


class SemanticRetriever:
    def __init__(self, corpus, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.corpus_embeddings = self.model.encode(corpus, show_progress_bar=True, convert_to_tensor=True)

    def score(self, query):
        q_embed = self.model.encode(query, convert_to_tensor=True)
        sims = util.cos_sim(q_embed, self.corpus_embeddings)[0].cpu().numpy()
        return sims


# -------------------------------------------------------------------
# Core Evaluation Workflow
# -------------------------------------------------------------------
def evaluate_models(data_path, gold_path, top_k=5):
    print(f"[INFO] Loading cleaned data from {data_path}")
    df = pd.read_csv(data_path)
    df.fillna("", inplace=True)

    print(f"[INFO] Loading gold relevance set: {gold_path}")
    gold = pd.read_csv(gold_path)

    # Build retrievers
    bm25 = BM25Retriever(df["clean_text"].tolist())
    tfidf = TFIDFRetriever(df["clean_text"].tolist())
    semantic = SemanticRetriever(df["clean_text"].tolist())

    # Prepare result table
    results = []

    for q_text in tqdm(gold["query"].unique(), desc="[Evaluating queries]"):
        subset = gold[gold["query"] == q_text]
        asin_to_rel = dict(zip(subset["parent_asin"], subset["relevance"]))

        # Build ground truth & prediction alignment
        y_true = np.array([asin_to_rel.get(a, 0) for a in df["parent_asin"]])

        # Get model scores
        bm25_scores = bm25.score(q_text)
        tfidf_scores = tfidf.score(q_text)
        sem_scores = semantic.score(q_text)

        results.append({
            "query": q_text,
            "bm25_p@k": precision_at_k(y_true, bm25_scores, top_k),
            "bm25_r@k": recall_at_k(y_true, bm25_scores, top_k),
            "bm25_map": mean_average_precision(y_true, bm25_scores),
            "tfidf_p@k": precision_at_k(y_true, tfidf_scores, top_k),
            "tfidf_r@k": recall_at_k(y_true, tfidf_scores, top_k),
            "tfidf_map": mean_average_precision(y_true, tfidf_scores),
            "sem_p@k": precision_at_k(y_true, sem_scores, top_k),
            "sem_r@k": recall_at_k(y_true, sem_scores, top_k),
            "sem_map": mean_average_precision(y_true, sem_scores),
        })

    res_df = pd.DataFrame(results)
    res_df.to_csv("data\\processed\\retrieval_evaluation_results.csv", index=False)
    print("\n=== SUMMARY RESULTS (avg across queries) ===")
    print(res_df.mean(numeric_only=True))
    print("\nDetailed results saved → data\\processed\\retrieval_evaluation_results.csv")


# -------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate BM25 / TF-IDF / Semantic retrieval")
    parser.add_argument("--data", required=True, help="Path to cleaned review CSV")
    parser.add_argument("--gold", required=True, help="Path to gold relevance CSV")
    parser.add_argument("--topk", type=int, default=5, help="Top-k cutoff for metrics")
    args = parser.parse_args()

    evaluate_models(args.data, args.gold, args.topk)


if __name__ == "__main__":
    main()