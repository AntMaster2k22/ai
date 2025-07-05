import pandas as pd
import numpy as np
import os

# --- Project Imports ---
from config import LABELED_DATA_CSV, AUTO_LABELED_CSV, URLS_FILE
from my_scraper import scrape_text_from_url
from model import model_pipeline, maybe_auto_label # CORRECTED IMPORT
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

def get_top_predictions(text, top_n=3):
    """
    Gets the top N predictions and their confidence scores from the loaded model.
    """
    # Use the globally loaded model pipeline
    pipeline = model_pipeline.pipeline
    if not pipeline:
        return [], None

    try:
        probs = pipeline.predict_proba([text])[0]
        # Get the indices of the top N probabilities
        top_indices = np.argsort(probs)[-top_n:][::-1]
        
        predictions = [(pipeline.classes_[i], probs[i]) for i in top_indices]
        suggested_label = predictions[0][0]
        
        return predictions, suggested_label
    except AttributeError:
        # Fallback for models like LinearSVC
        return [(model_pipeline.predict(text)[0], 1.0)], model_pipeline.predict(text)[0]


def curate_knowledge_session():
    """
    A guided session to scrape URLs, classify them, and interactively label them.
    """
    print("\n--- 🧠 Curate New Knowledge (Guided Mode) ---")
    
    # Check if the model is loaded
    if not model_pipeline.pipeline:
        print("[!] The model is not loaded. Please train a model first by selecting option '5' from the main menu.")
        return

    try:
        with open(URLS_FILE, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip()]
        if not urls:
            print(f"[!] The urls file at '{URLS_FILE}' is empty.")
            return
    except FileNotFoundError:
        print(f"[!] The urls file at '{URLS_FILE}' was not found.")
        return

    session_labeled_count = 0
    for url in urls:
        print(f"\n--- Scraping: {url} ---")
        text = scrape_text_from_url(url)
        if not text:
            print("[!] Could not extract text. Skipping.")
            continue
        
        predictions, suggested_label = get_top_predictions(text)
        
        print("\n--- AI Analysis ---")
        print(f"Text Snippet: \"{text[:300]}...\"")
        print(f"\nBased on my training, I think this is about '{suggested_label}'.")
        print("Here are my top predictions:")
        for i, (label, conf) in enumerate(predictions):
            print(f"{i+1}. {label} (Confidence: {conf:.2f})")
        
        print("\n--- Your Turn to Label ---")
        print("Choose the correct label:")
        for i, (label, _) in enumerate(predictions):
            print(f"  ({i+1}) Use '{label}'")
        if suggested_label:
             print(f"  (4) Confirm suggested '{suggested_label}'")
        print("  (n) Create a new label")
        print("  (s) Skip this article")
        
        choice = input("> ").lower()

        final_label = None
        if choice == 's':
            continue
        elif choice.isdigit() and 1 <= int(choice) <= len(predictions):
            final_label = predictions[int(choice)-1][0]
        elif choice == '4' and suggested_label:
            final_label = suggested_label
        elif choice == 'n':
            final_label = input("Enter new label name: ").strip().lower().replace(" ", "_")
        else:
            print("[!] Invalid choice. Skipping article.")
            continue

        if final_label:
            import csv
            # Use the robust maybe_auto_label logic, but force it by setting confidence to 1.0
            # This avoids duplicating file-writing logic.
            maybe_auto_label(text, final_label, 1.0) 
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