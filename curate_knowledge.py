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

def merge_auto_labeled_data():
    """Merges auto-labeled data into the main dataset and cleans up."""
    if not os.path.exists(AUTO_LABELED_CSV):
        return
    try:
        auto_df = pd.read_csv(AUTO_LABELED_CSV)
        if auto_df.empty:
            os.remove(AUTO_LABELED_CSV)
            return

        labeled_df = pd.read_csv(LABELED_DATA_CSV)
        combined_df = pd.concat([labeled_df, auto_df]).drop_duplicates(subset="text", keep='last')
        combined_df.to_csv(LABELED_DATA_CSV, index=False)

        print(f"\n[*] Merged {len(auto_df)} auto-labeled entries into {LABELED_DATA_CSV}.")
        os.remove(AUTO_LABELED_CSV)
    except Exception as e:
        print(f"[!] Error merging auto-labeled data: {e}")

def get_top_predictions(pipeline, text, top_n=3):
    """
    Gets the top N predictions and their confidence scores for a given text.
    Handles both 'predict_proba' and 'decision_function' models.
    """
    try:
        probs = pipeline.predict_proba([text])[0]
        top_indices = np.argsort(probs)[-top_n:][::-1]
        top_labels = pipeline.classes_[top_indices]
        top_confidences = probs[top_indices]
    except AttributeError:
        decision_values = pipeline.decision_function([text])[0]
        top_indices = np.argsort(decision_values)[-top_n:][::-1]
        top_labels = pipeline.classes_[top_indices]
        top_confidences = (decision_values[top_indices] - decision_values.min()) / (decision_values.max() - decision_values.min()) if (decision_values.max() - decision_values.min()) > 0 else np.ones_like(decision_values[top_indices])
    return list(zip(top_labels, top_confidences))

def find_wiki_links(topic):
    """
    Finds related Wikipedia links from a topic page.
    """
    search_url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
    try:
        response = requests.get(search_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'lxml')
        content_div = soup.find(id="mw-content-text")
        if not content_div:
            return [search_url]
        links = set()
        for a_tag in content_div.find_all('a', href=True):
            href = a_tag['href']
            if href.startswith('/wiki/') and ':' not in href:
                links.add(f"https://en.wikipedia.org{href}")
        print(f"[*] Found {len(links)} potential articles related to '{topic}'.")
        return list(links) if links else [search_url]
    except requests.RequestException as e:
        print(f"[!] Could not fetch Wikipedia page for '{topic}': {e}")
        return []

def clean_wiki_title(url):
    """Extracts and cleans a label from a Wikipedia URL."""
    try:
        title = url.split('/wiki/')[1]
        # Remove parenthetical extras like (programming_language)
        title = re.sub(r'_\(.*\)', '', title)
        title = title.replace('_', ' ').strip()
        return title.lower()
    except IndexError:
        return None

def curate_knowledge_session():
    """A powerful, continuous session for curating new knowledge."""
    print("\n--- 🚀 AI Knowledge Curation Mode ---")
    print("Enter a broad Wikipedia topic (e.g., 'List of programming languages', 'Roman Emperors').")
    
    try:
        pipeline = load_pipeline()
    except Exception as e:
        print(f"[!] Failed to load model: {e}. Please run learn.py first.")
        return

    while True:
        main_topic = input("\nEnter a new topic to explore, or type 'quit' to exit: ")
        if main_topic.lower() == 'quit':
            break
        
        urls_to_process = find_wiki_links(main_topic)
        if not urls_to_process:
            continue

        session_labeled_count = 0
        for i, url in enumerate(urls_to_process):
            print(f"\n--- Processing article {i+1}/{len(urls_to_process)}: {url} ---")
            
            text = scrape_text_from_url(url)
            if not text or len(text) < 200:
                print("[!] Scraped text is too short. Skipping.")
                continue

            predictions = get_top_predictions(pipeline, text, top_n=3)
            suggested_label = clean_wiki_title(url)

            print(f"Snippet: {text[:700]}...")
            print("\n--- What is the correct label? ---")
            for idx, (label, conf) in enumerate(predictions):
                print(f"{idx+1}. {label} (Confidence: {conf:.2f})")
            
            # --- THE KEY UPGRADE ---
            # Offer the cleaned article title as an option
            if suggested_label:
                print(f"4. Use article title: '{suggested_label}' (Recommended for new specific topics)")
            
            print("n. Type a different new label manually")
            print("s. Skip this article")
            print("q. Quit this topic")

            choice = input("> ").lower().strip()

            if choice == 'q': break
            if choice == 's': continue

            final_label = ""
            if choice.isdigit() and 1 <= int(choice) <= 3:
                final_label = predictions[int(choice)-1][0]
            elif choice == '4' and suggested_label:
                final_label = suggested_label
            elif choice == 'n':
                final_label = input("Enter new label name: ").strip().lower().replace(" ", "_")
            else:
                print("[!] Invalid choice. Skipping article.")
                continue

            if final_label:
                snippet_to_save = '"' + text.replace('"', '""') + '"'
                new_data = {'text': [snippet_to_save], 'label': [final_label]}
                new_df = pd.DataFrame(new_data)
                new_df.to_csv(LABELED_DATA_CSV, mode='a', header=not os.path.exists(LABELED_DATA_CSV), index=False)
                print(f"✅ Saved with label '{final_label}'.")
                session_labeled_count += 1

        if session_labeled_count > 0:
            print(f"\n--- Session Summary ---")
            print(f"You added {session_labeled_count} new high-quality labels.")
            retrain_choice = input("Would you like to retrain the model with this new data now? (y/n): ").lower()
            if retrain_choice == 'y':
                print("\n[*] Merging any other auto-labeled data first...")
                merge_auto_labeled_data()
                print("\n[*] Retraining model...")
                train_model()
        else:
            print("\nNo new labels were added in this session.")

if __name__ == "__main__":
    curate_knowledge_session()