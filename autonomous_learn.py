import pandas as pd
import numpy as np
import sys
import os
import requests
import re
from bs4 import BeautifulSoup

# --- Project Imports ---
from config import LABELED_DATA_CSV, AUTO_LABELED_CSV
from my_scraper import scrape_text_from_url
from model import load_pipeline
from learn import train_model

# --- Configuration for Autonomous Mode ---
# The AI will only auto-label if its confidence is above this threshold.
# Start high (0.95) to ensure high quality. You can lower it as the AI gets smarter.
MIN_CONFIDENCE_TO_AUTO_LABEL = 0.95

def get_best_prediction(pipeline, text):
    """Gets the single best prediction and its confidence score."""
    try:
        probs = pipeline.predict_proba([text])[0]
        confidence = np.max(probs)
        prediction = pipeline.classes_[np.argmax(probs)]
    except AttributeError:
        decision_values = pipeline.decision_function([text])[0]
        prediction = pipeline.predict([text])[0]
        # Normalize for a confidence score
        if (decision_values.max() - decision_values.min()) > 0:
            confidence = (np.max(decision_values) - decision_values.min()) / (decision_values.max() - decision_values.min())
        else:
            confidence = 1.0
    return prediction, confidence

def find_wiki_links(topic):
    """Finds related Wikipedia links from a topic page."""
    search_url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
    try:
        response = requests.get(search_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'lxml')
        content_div = soup.find(id="mw-content-text")
        if not content_div: return [search_url]
        links = {f"https://en.wikipedia.org{a['href']}" for a in content_div.find_all('a', href=True) if a['href'].startswith('/wiki/') and ':' not in a['href']}
        print(f"[*] Found {len(links)} potential articles related to '{topic}'.")
        return list(links) if links else [search_url]
    except requests.RequestException as e:
        print(f"[!] Could not fetch Wikipedia page for '{topic}': {e}")
        return []

def run_autonomous_session():
    """
    An autonomous workflow for the AI to learn on its own.
    It finds, scrapes, and labels articles without user intervention,
    using a confidence threshold as a safety measure.
    """
    print("\n--- 🤖 Autonomous Learning Mode ---")
    
    try:
        pipeline = load_pipeline()
        print("[*] Model loaded successfully.")
    except Exception as e:
        print(f"[!] Failed to load model: {e}. Please run learn.py first.")
        return

    main_topic = input("Enter a broad starting topic for the AI to explore (e.g., 'History of science'): ")
    if not main_topic:
        return

    urls_to_process = find_wiki_links(main_topic)
    if not urls_to_process:
        return

    print(f"[*] The AI will now process {len(urls_to_process)} articles.")
    print(f"[*] It will only save new data if its confidence is above {MIN_CONFIDENCE_TO_AUTO_LABEL * 100}%.")

    newly_labeled_data = []
    for i, url in enumerate(urls_to_process):
        # Basic progress indicator
        print(f"\r[*] Processing article {i+1}/{len(urls_to_process)}...", end="")
        
        text = scrape_text_from_url(url, silent=True) # Run scraper in silent mode
        if not text or len(text) < 500: # Skip short articles
            continue

        prediction, confidence = get_best_prediction(pipeline, text)

        if confidence >= MIN_CONFIDENCE_TO_AUTO_LABEL:
            snippet_to_save = '"' + text.replace('"', '""') + '"'
            newly_labeled_data.append({'text': snippet_to_save, 'label': prediction})
    
    print("\n--- Autonomous Session Complete ---")

    if not newly_labeled_data:
        print("[*] The AI didn't find any new information it was confident enough to add.")
        return

    print(f"✅ The AI autonomously identified and labeled {len(newly_labeled_data)} new articles.")
    
    # Save the high-confidence data to the auto-labeled file
    new_df = pd.DataFrame(newly_labeled_data)
    new_df.to_csv(AUTO_LABELED_CSV, mode='a', header=not os.path.exists(AUTO_LABELED_CSV), index=False)

    print(f"[*] This new data has been saved to '{AUTO_LABELED_CSV}'.")
    print("[*] You can run the 'Curation Mode' or 'learn.py' to merge it and retrain the model.")
    
if __name__ == "__main__":
    run_autonomous_session()