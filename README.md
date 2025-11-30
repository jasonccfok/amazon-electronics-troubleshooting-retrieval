# Amazon Electronics Troubleshooting Retrieval System  
CS 410 - Text Information Systems  
Team Member & Project Coordinator: Jason Fok (ccfok2)  

---

## Overview

The Amazon Electronics Troubleshooting Retrieval System is designed to help users find solutions to common problems with electronic products by mining Amazon review data. Users can describe their issue in natural language (e.g., “Bluetooth keeps disconnecting” or “screen flickering after update”), and the system retrieves relevant reviews where others have reported similar issues or offered fixes.


## Project Structure
```
project_root/
├── preprocessing/
│   ├── load_raw_data.py                 # Loads and merges Amazon reviews & product metadata
│   ├── text_cleaning.py                 # Cleans and normalizes text (HTML removal, lowercasing, etc.)
│   └── exploratory_data_analysis.ipynb  # Exploratory Data Analysis (EDA) on merged & cleaned datasets
│
├── retrieval/
│   ├── bm25_retrieval.py                # Lexical retrieval using BM25 algorithm
│   └── semantic_retrieval.py            # Semantic retrieval using transformer-based embeddings
│
├── evaluation/
│   └── retrieval_evaluation.py          # Evaluates retrieval performance (Precision@K, Recall@K, etc.)
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
```

## Demo Commands (Windows Command Prompt Examples)

### BM25 Lexical Retrieval

cmd
python "2. retrieval\bm25_retrieval.py" ^
    --data "data\processed\reviews_clean.csv" ^
    --query "tripod screw loose problem" ^
    --topk 5


Output Example:

================= QUERY RESULTS =================
Query: tripod screw loose problem

------ BM25 Top Results ------
1. parent_asin: B0BZB56D2J | Score: 16.0709
   Review (raw): Loose screw inside, Bulky but very useful! The design is very bulky but it's very useful feature to clamp into place. quality feels nice. Mine has 1 problem which does worries me a little: when I shake it a little you can tell there is a screw loose inside. this is a potential cause of a short circuit. But because once is set in place I don't move it anymore it doesn't worry me as much. Tripp Lite 3 Outlet Surge Protector Power Strip with Desk Clamp, 10ft. Cord, 510 Joules, 2 USB Charging Ports, Black, $20K Insurance & (TLP310USBC)

------ TF-IDF Top Results ------
1. parent_asin: B0BZB56D2J | Score: 0.1722
   Review (raw): Loose screw inside, Bulky but very useful! The design is very bulky but it's very useful feature to clamp into place. quality feels nice. Mine has 1 problem which does worries me a little: when I shake it a little you can tell there is a screw loose inside. this is a potential cause of a short circuit. But because once is set in place I don't move it anymore it doesn't worry me as much. Tripp Lite 3 Outlet Surge Protector Power Strip with Desk Clamp, 10ft. Cord, 510 Joules, 2 USB Charging Ports, Black, $20K Insurance & (TLP310USBC)


### Semantic Retrieval (Transformer-based)

cmd
python "2. retrieval\semantic_retrieval.py" ^
    --data "data\processed\reviews_clean.csv" ^
    --query "phone adapter screw problem" ^
    --topk 5 ^
    --model "all-MiniLM-L6-v2"


Output Example:

================= QUERY RESULTS =================
Query: phone adapter screw problem

------ Semantic (SentenceTransformer) Top Results ------
1. parent_asin: B006C13X4Q | Score: 0.6141
   Review (raw): wont work with adapter I bought this as a cute gag gift for my dad for Christmas. He doesn't have an iPhone so I bought an adapter to use with it and it didn't work for him AT ALL! It was so upsetting on Christmas day. It did however work on the iPhone, just not on his with an adapter. SANOXY Retro Cell Phone Handset
