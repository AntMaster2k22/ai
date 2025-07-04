# config.py

import os

# --- Directory Paths ---
DATA_DIR = "data"

# Ensure the data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# --- File Paths ---
URLS_FILE = os.path.join(DATA_DIR, "urls.txt")
TOPICS_FILE = os.path.join(DATA_DIR, "topics.txt")
LABELED_DATA_CSV = os.path.join(DATA_DIR, "labeled_data.csv")
AUTO_LABELED_CSV = os.path.join(DATA_DIR, "auto_labeled_data.csv")
MODEL_PATH = os.path.join(DATA_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(DATA_DIR, "vectorizer.pkl")
INDEX_PATH = os.path.join(DATA_DIR, "memory.index")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.pkl")

# --- Model & Embedding Settings ---
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
VECTOR_DIMENSION = 384  # Based on the chosen embedding model

# --- Learning & Classification Settings ---
CONFIDENCE_THRESHOLD = 0.85  # Threshold for auto-labeling new data