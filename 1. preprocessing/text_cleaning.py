"""
text_cleaning.py

Cleans and normalizes combined review texts produced by load_raw_data.py,
and saves BOTH raw and cleaned versions.

This script:
 1. Load merged CSV (parent_asin, combined_text)
 2. Normalize text (lowercase, remove URLs/punctuation/extra spaces)
 3. Optionally remove stopwords
 4. Lemmatize words
 5. Save both raw and cleaned text to reviews_clean.csv

Usage (Windows command prompt example):
> python "1. preprocessing\\text_cleaning.py" ^
    --input "data\\processed\\electronics_merged.csv" ^
    --output "data\\processed\\reviews_clean.csv"
"""

import argparse
import pandas as pd
import re
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# -------------------------------------------------------------------
# NLTK Resource Setup
# -------------------------------------------------------------------
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# -------------------------------------------------------------------
# Core Cleaning Functions
# -------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Normalize and remove unwanted characters from text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)   # remove URLs
    text = re.sub(r"<.*?>", " ", text)                     # remove HTML
    text = re.sub(r"[^a-z0-9\s]", " ", text)               # remove punctuation/special chars
    text = re.sub(r"\s+", " ", text)                       # collapse whitespace
    return text.strip()


def remove_stopwords(text: str) -> str:
    """Remove English stopwords."""
    tokens = word_tokenize(text)
    filtered = [t for t in tokens if t not in STOPWORDS]
    return " ".join(filtered)


def lemmatize_as_verb(text: str) -> str:
    """Lemmatize all words, assuming they are verbs (no POS tagging)."""
    tokens = word_tokenize(text)
    lemmas = [LEMMATIZER.lemmatize(t, pos="v") for t in tokens]
    return " ".join(lemmas)

# -------------------------------------------------------------------
# Main Preprocessing Workflow
# -------------------------------------------------------------------
def preprocess_reviews(input_path: str, output_path: str, remove_stops: bool = True):
    """Clean, normalize, and lemmatize review text, retaining raw text for display."""
    print(f"[INFO] Loading merged data from {input_path}")
    df = pd.read_csv(input_path)
    if "combined_text" not in df.columns:
        raise ValueError("[ERROR] Input file must include a 'combined_text' column.")

    # Keep the raw uncleaned version for later printing in retrieval
    df["review_text"] = df["combined_text"]

    print("[INFO] Cleaning text ...")
    tqdm.pandas(desc="[Cleaning]")
    df["clean_text"] = df["combined_text"].progress_apply(clean_text)

    if remove_stops:
        print("[INFO] Removing stopwords ...")
        df["clean_text"] = df["clean_text"].progress_apply(remove_stopwords)

    print("[INFO] Lemmatizing text (verb-biased) ...")
    df["clean_text"] = df["clean_text"].progress_apply(lemmatize_as_verb)

    # Keep relevant columns
    df_out = df[["parent_asin", "review_text", "clean_text"]]
    df_out.to_csv(output_path, index=False, encoding="utf-8")

    print(f"[INFO] Cleaned data saved → {output_path}")
    print(f"[INFO] Total records: {len(df_out)}")
    print(f"[INFO] Sample:\n{df_out.head(2)}")


# -------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and normalize Amazon review text (retain raw + clean).")
    parser.add_argument("--input", required=True, help="Path to merged CSV (e.g., data\\processed\\electronics_merged.csv)")
    parser.add_argument("--output", required=True, help="Output CSV path for cleaned text (e.g., data\\processed\\reviews_clean.csv)")
    parser.add_argument("--no-stopwords", action="store_true", help="Skip stopword removal if specified.")
    args = parser.parse_args()

    preprocess_reviews(args.input, args.output, remove_stops=not args.no_stopwords)