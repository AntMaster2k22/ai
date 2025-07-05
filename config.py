# config.py

import os

# --- Directory Paths ---
DATA_DIR = "data"

# Ensure the data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# --- API & Service Keys ---
SCRAPER_API_KEY = '418d488bcc86115abc22091769d23769'
ABSTRACT_API_KEY  = "077be81914c74fc4b2ec1ecfd5d5023c"
CRITIQUE_API_KEY = "0bKPR3MBmXYvo5DzN25-fn_M2v3pSlqaOc1Rs3L888M"
SERP_API_KEY    = "61f9d3cf6fe487208311185efb4918659619bd4a3f9929fadb7859c96966b819"
RAPIDAPI_KEY    = "55c764dbb0mshdc473c8fb8f8996p17115cjsn289b47b89222"


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

# --- Harvester Settings ---
HARVESTER_BLACKLISTED_DOMAINS = {
    'youtube.com', 'facebook.com', 'twitter.com', 'instagram.com', 'reddit.com', 'amazon.com',
    'linkedin.com', 'pinterest.com', 'tiktok.com', 'wikipedia.org', 'forbes.com', 'quora.com',
    'stackoverflow.com', 'medium.com', 'github.com', 'google.com', 'microsoft.com', 'apple.com',
    'telegram.org', 'discord.com', 'archive.org', 'slideshare.net', 'scribd.com', 'researchgate.net',
    'cnet.com', 'techcrunch.com', 'wired.com', 'arstechnica.com', 'imdb.com', 'rottentomatoes.com'
}

HARVESTER_BLACKLISTED_EXTENSIONS = {
    '.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip', '.rar', '.exe', '.mp3',
    '.mp4', '.avi', '.mov', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx'
}

HARVESTER_MIN_CONTENT_WORDS = 200