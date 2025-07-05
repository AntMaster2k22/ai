import pandas as pd
import os
import requests
from bs4 import BeautifulSoup
import time
import logging # Import logging

# --- Project Imports ---
from config import AUTO_LABELED_CSV, MIN_CONFIDENCE_TO_AUTO_LABEL, logger # Import MIN_CONFIDENCE_TO_AUTO_LABEL and logger from config
from my_scraper import scrape_text_from_url
from model import predict_with_confidence, maybe_auto_label # CORRECTED IMPORT
from learn import train_model

# Use the logger configured in config.py
logger = logging.getLogger(__name__)

# MIN_CONFIDENCE_TO_AUTO_LABEL is now in config.py, so remove the local definition
# MIN_CONFIDENCE_TO_AUTO_LABEL = 0.95 # REMOVE THIS LINE

def find_wiki_links(topic):
    """Finds related Wikipedia links for a given topic."""
    logger.info(f"\n[*] Searching for related articles on Wikipedia for topic: '{topic}'...")
    try:
        search_url = f"https://en.wikipedia.org/w/index.php?search={topic.replace('_', '+')}"
        response = requests.get(search_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        links = set()
        # Find links within the main content area
        for a in soup.select('div.mw-search-result-heading a, ul li a'):
            href = a.get('href')
            if href and href.startswith('/wiki/') and ':' not in href:
                links.add(f"https://en.wikipedia.org{href}")

        logger.info(f"[+] Found {len(links)} potential articles for topic '{topic}'.")
        return list(links)[:15] # Limit to 15 to keep sessions manageable
    except requests.exceptions.Timeout:
        logger.warning(f"[x] Timeout occurred while searching Wikipedia for '{topic}'.")
        return []
    except requests.exceptions.HTTPError as e:
        logger.warning(f"[x] HTTP Error {e.response.status_code} while searching Wikipedia for '{topic}': {e}")
        return []
    except requests.exceptions.ConnectionError:
        logger.warning(f"[x] Connection Error while searching Wikipedia for '{topic}'.")
        return []
    except requests.exceptions.RequestException as e:
        logger.warning(f"[x] An unknown Request Error occurred while searching Wikipedia for '{topic}': {e}")
        return []
    except Exception as e:
        logger.error(f"[x] An unexpected error occurred in find_wiki_links for '{topic}': {e}")
        return []

def run_autonomous_session():
    """
    Runs an autonomous learning session.
    The AI searches for articles, scrapes them, and auto-labels them if confident.
    """
    logger.info("\n--- 🧠 Autonomous Learning Session Starting ---")

    # In a real autonomous system, topics would likely come from a dynamic source
    # For now, we use a hardcoded list, or perhaps load from topics.txt if available.
    # Assuming TOPICS_FILE can be used here.
    topics = []
    try:
        from config import TOPICS_FILE # Assuming TOPICS_FILE is in config.py
        if os.path.exists(TOPICS_FILE):
            with open(TOPICS_FILE, 'r', encoding='utf-8') as f:
                topics = [line.strip() for line in f if line.strip()]
            if not topics:
                logger.warning(f"No topics found in {TOPICS_FILE}. Using default topics.")
                topics = ["artificial intelligence", "machine learning", "neural networks", "data science"]
        else:
            logger.warning(f"{TOPICS_FILE} not found. Using default topics for autonomous session.")
            topics = ["artificial intelligence", "machine learning", "neural networks", "data science"]
    except Exception as e:
        logger.error(f"Error loading topics from {TOPICS_FILE}: {e}. Using default topics.")
        topics = ["artificial intelligence", "machine learning", "neural networks", "data science"]


    urls_to_process = set()
    for topic in topics:
        found_links = find_wiki_links(topic)
        urls_to_process.update(found_links)

    if not urls_to_process:
        logger.info("[*] No new URLs found to process in this autonomous session.")
        return

    logger.info(f"\n[*] AI will now attempt to learn from {len(urls_to_process)} articles.")
    logger.info(f"[*] It will only save new data if its confidence is above {MIN_CONFIDENCE_TO_AUTO_LABEL * 100:.2f}%.")

    newly_labeled_count = 0
    for i, url in enumerate(urls_to_process):
        logger.info(f"Processing article {i+1}/{len(urls_to_process)}: {url[:70]}...") # Changed from print to logger.info

        text = scrape_text_from_url(url, silent=True)
        if not text or len(text.split()) < 500: # Use split() for word count, not just len(text)
            logger.warning(f"Skipping URL {url} due to failed text extraction or insufficient content length.")
            continue

        try:
            # Use the corrected, globally accessible function
            prediction, confidence = predict_with_confidence(text)

            if confidence >= MIN_CONFIDENCE_TO_AUTO_LABEL:
                maybe_auto_label(text, prediction, confidence)
                newly_labeled_count += 1
                logger.info(f"✅ Auto-labeled '{prediction}' (conf: {confidence:.2f}) for URL {url[:70]}...")
            else:
                logger.info(f"[*] Skipped URL {url[:70]}... - Confidence ({confidence:.2f}) below threshold ({MIN_CONFIDENCE_TO_AUTO_LABEL}).")
        except Exception as e:
            logger.error(f"[x] Error during prediction/auto-labeling for URL {url}: {e}", exc_info=True)
            continue

    logger.info("--- Autonomous Session Complete ---")

    if newly_labeled_count == 0:
        logger.info("[*] The AI didn't find any new information it was confident enough to add in this session.")
        return

    logger.info(f"✅ The AI autonomously identified and labeled {newly_labeled_count} new articles.")

    # Remove the interactive retraining prompt. In an autonomous system,
    # the main orchestration loop (in main.py) will handle calls to data_doctor,
    # curate_knowledge, and eventual retraining if conditions are met.
    # The curate_knowledge.py's autonomous mode now handles merging and re-training
    # after its own curation session, which is downstream of this.

if __name__ == '__main__':
    run_autonomous_session()