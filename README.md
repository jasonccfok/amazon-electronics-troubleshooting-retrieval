# Amazon Electronics Troubleshooting Retrieval System  
CS 410 - Text Information Systems  
Team Member & Project Coordinator: Jason Fok (ccfok2)  

---

## Overview

The Amazon Electronics Troubleshooting Retrieval System is designed to help users find solutions to common problems with electronic products by mining Amazon review data. Users can describe their issue in natural language (e.g., “Bluetooth keeps disconnecting” or “screen flickering after update”), and the system retrieves relevant reviews where others have reported similar issues or offered fixes.


## Project Structure

project_root/
├── preprocessing/
│   ├── load_raw_data.py         # Loads and merges Amazon reviews & product metadata
│   └── text_cleaning.py         # Cleans and normalizes text (HTML removal, lowercasing, etc.)
│
├── retrieval/
│   ├── bm25_retrieval.py        # Lexical retrieval using BM25 algorithm
│   └── semantic_retrieval.py    # Semantic retrieval using transformer-based embeddings
│
├── evaluation/
│   └── retrieval_evaluation.py  # Evaluates retrieval performance (Precision@K, Recall@K, etc.)
│
├── data/
│   ├── raw/
│   │   ├── Electronics.jsonl.gz         # Raw review dataset from Amazon
│   │   └── meta_Electronics.jsonl.gz    # Metadata such as product titles and ASINs
│   │
│   ├── processed/
│   │   ├── electronics_merged.csv       # Combined reviews and metadata after preprocessing
│   │   ├── review_embeddings.pkl        # Precomputed embeddings for semantic retrieval
│   │   └── reviews_clean.csv            # Cleaned text ready for BM25 or embedding generation
│   │
│   ├── groundtruth/
│   │   ├── gold_set.csv                 # Manually or semi-automatically labeled relevance set
│   │   └── retrieval_evaluation_results.csv # Evaluation metrics comparing retrieval models
│
├── README.md
└── requirements.txt