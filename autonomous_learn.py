import pandas as pd
import os
import requests
from bs4 import BeautifulSoup
import time

# --- Project Imports ---
from config import AUTO_LABELED_CSV
from my_scraper import scrape_text_from_url
from model import predict_with_confidence # CORRECTED IMPORT
from learn import train_model

# --- Configuration for Autonomous Mode ---
MIN_CONFIDENCE_TO_AUTO_LABEL = 0.95

def find_wiki_links(topic):
    """Finds related Wikipedia links for a given topic."""
    print(f"\n[*] Searching for related articles on Wikipedia for topic: '{topic}'...")
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
        
        print(f"[+] Found {len(links)} potential articles.")
        return list(links)[:15] # Limit to 15 to keep sessions manageable
    except requests.RequestException as e:
        print(f"[!] Could not fetch Wikipedia links: {e}")
        return []

def run_autonomous_session():
    """
    An autonomous session where the AI finds and learns from new articles.
    """
    print("\n--- 🤖 Autonomous Learning Session ---")
    
    # 1. Get existing labels from the model
    from model import model_pipeline
    if not model_pipeline.pipeline:
        print("[!] Model not loaded. Please train a model first.")
        return
        
    existing_labels = model_pipeline.pipeline.classes_
    print(f"[*] Current knowledge topics: {', '.join(existing_labels)}")
    
    # 2. Find new URLs to process
    urls_to_process = []
    for label in existing_labels:
        urls_to_process.extend(find_wiki_links(label))
        time.sleep(1) # Be respectful to Wikipedia's servers
    
    if not urls_to_process:
        print("[!] Could not find any new articles to learn from.")
        return
        
    print(f"\n[*] AI will now attempt to learn from {len(urls_to_process)} articles.")
    print(f"[*] It will only save new data if its confidence is above {MIN_CONFIDENCE_TO_AUTO_LABEL * 100}%.")

    newly_labeled_count = 0
    for i, url in enumerate(urls_to_process):
        print(f"\r[*] Processing article {i+1}/{len(urls_to_process)}: {url[:70]}...", end="")
        
        text = scrape_text_from_url(url, silent=True)
        if not text or len(text) < 500: # Skip short articles
            continue

        # Use the corrected, globally accessible function
        prediction, confidence = predict_with_confidence(text)

        if confidence >= MIN_CONFIDENCE_TO_AUTO_LABEL:
            from model import maybe_auto_label
            maybe_auto_label(text, prediction, confidence)
            newly_labeled_count += 1
    
    print("\n--- Autonomous Session Complete ---")

    if newly_labeled_count == 0:
        print("[*] The AI didn't find any new information it was confident enough to add.")
        return

    print(f"✅ The AI autonomously identified and labeled {newly_labeled_count} new articles.")
    
    retrain_choice = input("Would you like to merge this new data and retrain the model now? (y/n): ").lower()
    if retrain_choice == 'y':
        from merge_labeledcsv import merge_files
        print("\n[*] Merging auto-labeled data...")
        merge_files()
        print("\n[*] Retraining model...")
        train_model()