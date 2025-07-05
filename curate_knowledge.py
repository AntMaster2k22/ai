import pandas as pd
import numpy as np
import os
import logging # Import logging
import csv # Import csv for quoting in read/write

# --- Project Imports ---
from config import LABELED_DATA_CSV, AUTO_LABELED_CSV, URLS_FILE, MIN_CONFIDENCE_TO_AUTO_LABEL # Ensure MIN_CONFIDENCE_TO_AUTO_LABEL is imported
from my_scraper import scrape_text_from_url
from model import model_pipeline, maybe_auto_label # CORRECTED IMPORT
from learn import train_model

# Configure logging (similar to data_doctor.py, or assume global config)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def merge_auto_labeled_data():
    """Merges auto-labeled data into the main dataset and cleans up."""
    if not os.path.exists(AUTO_LABELED_CSV):
        logger.info(f"No auto-labeled data file found at {AUTO_LABELED_CSV}.")
        return
    try:
        auto_df = pd.read_csv(AUTO_LABELED_CSV, quoting=csv.QUOTE_ALL, escapechar='\\')
        if auto_df.empty:
            logger.info("Auto-labeled data file is empty. Removing it.")
            os.remove(AUTO_LABELED_CSV)
            return

        # Ensure main labeled data file exists, if not, create an empty DataFrame
        if os.path.exists(LABELED_DATA_CSV):
            labeled_df = pd.read_csv(LABELED_DATA_CSV, quoting=csv.QUOTE_ALL, escapechar='\\')
        else:
            labeled_df = pd.DataFrame(columns=['text', 'label', 'confidence']) # Assuming these columns

        combined_df = pd.concat([labeled_df, auto_df]).drop_duplicates(subset="text", keep='last')
        combined_df.to_csv(LABELED_DATA_CSV, index=False, quoting=csv.QUOTE_ALL, escapechar='\\')

        logger.info(f"[*] Merged {len(auto_df)} auto-labeled entries into {LABELED_DATA_CSV}.")
        os.remove(AUTO_LABELED_CSV)
    except pd.errors.EmptyDataError:
        logger.warning(f"Auto-labeled CSV '{AUTO_LABELED_CSV}' is empty. Removing it.")
        if os.path.exists(AUTO_LABELED_CSV):
            os.remove(AUTO_LABELED_CSV)
    except FileNotFoundError as e:
        logger.error(f"[x] File not found during merge: {e}")
    except Exception as e:
        logger.error(f"[x] Error merging auto-labeled data: {e}")

def get_top_predictions(text, top_n=3):
    """
    Gets the top N predictions and their probabilities from the model pipeline.
    Assumes model_pipeline returns a list of (label, probability) tuples.
    """
    try:
        # Assuming model_pipeline handles input correctly and returns predictions
        predictions_raw = model_pipeline.predict_proba([text])[0] # Assuming this structure
        classes = model_pipeline.classes_ # Assuming model_pipeline has .classes_ attribute

        # Combine predictions with class names and sort by probability
        predictions_with_probs = sorted(zip(classes, predictions_raw), key=lambda x: x[1], reverse=True)

        # Get the top N predictions
        top_predictions = predictions_with_probs[:top_n]

        # Get the suggested label for auto-labeling (usually the top one)
        suggested_label = top_predictions[0][0] if top_predictions and top_predictions[0][1] >= MIN_CONFIDENCE_TO_AUTO_LABEL else None
        
        return top_predictions, suggested_label
    except Exception as e:
        logger.error(f"[x] Error getting top predictions: {e}")
        return [], None # Return empty list and None on error


def run_curation_session(autonomous_mode: bool = False, min_auto_label_confidence: float = MIN_CONFIDENCE_TO_AUTO_LABEL):
    """
    Manages the knowledge curation session, either interactively or autonomously.

    Args:
        autonomous_mode (bool): If True, the script runs non-interactively
                                and applies predefined clean-up rules.
        min_auto_label_confidence (float): Minimum confidence for auto-labeling in autonomous mode.
    """
    logger.info(f"\n--- 🧠 Knowledge Curation Session Starting (Autonomous Mode: {autonomous_mode}) ---")

    session_labeled_count = 0
    # In a real scenario, you'd fetch unlabeled articles from a queue or a file
    # For this example, let's assume we have a way to get articles to curate
    # This might come from my_scraper.py output that isn't yet in labeled_data.csv
    # For now, let's simulate fetching some content to be curated if not in autonomous mode
    
    # In a fully autonomous system, this part would be fed by new scraped content
    # For demonstration, let's assume `urls_to_curate` comes from a source
    # For example, URLs that have been scraped but not yet processed/labeled.
    # We can use the URLS_FILE as a source for this, assuming they are raw URLs
    
    urls_to_curate = []
    if os.path.exists(URLS_FILE):
        with open(URLS_FILE, 'r', encoding='utf-8') as f:
            urls_to_curate = [line.strip() for line in f if line.strip()]
        logger.info(f"Found {len(urls_to_curate)} URLs to potentially curate.")
    else:
        logger.warning(f"No {URLS_FILE} found. Cannot curate new knowledge without URLs.")
        return # Exit if no URLs to curate


    for url in urls_to_curate[:20]: # Process a small batch for demonstration
        logger.info(f"Processing URL: {url}")
        text = scrape_text_from_url(url, silent=True) # Use my_scraper for actual text
        if not text:
            logger.warning(f"Skipping URL {url} due to failed text extraction.")
            continue

        predictions, suggested_label = get_top_predictions(text)

        if not predictions:
            logger.warning(f"Skipping URL {url} due to no predictions from model.")
            continue

        if not autonomous_mode:
            # Interactive flow
            print("\n--- Article to Curate ---")
            print(f"Source URL: {url}")
            print(f"Text snippet: {text[:500]}...")
            print("Suggested Labels:")
            for i, (label, prob) in enumerate(predictions):
                print(f"{i+1}. {label} ({prob:.2f})")
            if suggested_label:
                print(f"4. Suggested by Auto-Labeler: {suggested_label}")
            print("Enter new label [n], [s]kip article, or choose a number: ")

            choice = input("> ").lower()

            final_label = None
            if choice == 's':
                logger.info(f"Skipped URL {url} by user choice.")
                continue
            elif choice.isdigit() and 1 <= int(choice) <= len(predictions):
                final_label = predictions[int(choice)-1][0]
                logger.info(f"User chose existing label: {final_label} for URL {url}.")
            elif choice == '4' and suggested_label:
                final_label = suggested_label
                logger.info(f"User accepted auto-suggested label: {final_label} for URL {url}.")
            elif choice == 'n':
                final_label = input("Enter new label name: ").strip().lower().replace(" ", "_")
                if final_label:
                    logger.info(f"User created new label: {final_label} for URL {url}.")
                else:
                    logger.warning(f"No new label provided for URL {url}. Skipping.")
                    continue
            else:
                logger.warning(f"[!] Invalid choice for URL {url}. Skipping article.")
                continue

            if final_label:
                # Use the robust maybe_auto_label logic, but force it by setting confidence to 1.0
                maybe_auto_label(text, final_label, 1.0)
                logger.info(f"✅ Saved with label '{final_label}' for URL {url}.")
                session_labeled_count += 1

        else: # Autonomous mode logic
            # Auto-label if top prediction confidence is high enough
            top_pred_label = predictions[0][0]
            top_pred_conf = predictions[0][1]

            if top_pred_conf >= min_auto_label_confidence:
                maybe_auto_label(text, top_pred_label, top_pred_conf)
                logger.info(f"✅ Auto-labeled '{top_pred_label}' (conf: {top_pred_conf:.2f}) for URL {url}.")
                session_labeled_count += 1
            else:
                logger.info(f"[*] Skipped auto-labeling for URL {url}. Top confidence ({top_pred_conf:.2f}) below threshold ({min_auto_label_confidence}).")
                # Optionally, save to a 'needs_manual_review.csv' for later
                # Or simply discard/log if strict auto-curation is desired.

    if session_labeled_count > 0:
        logger.info(f"\n--- Session Summary ---")
        logger.info(f"Processed {session_labeled_count} new high-quality labels.")
        
        # In autonomous mode, automatically merge and consider retraining
        if autonomous_mode:
            logger.info("[*] Merging any other auto-labeled data first in autonomous mode...")
            merge_auto_labeled_data()
            logger.info("[*] Considering model retraining after autonomous curation.")
            # Here you might add logic to call train_model if specific criteria are met
            # e.g., if enough new data has been added, or if model performance has degraded.
            train_model() # Placeholder - a real autonomous system would have more intelligent retraining triggers
        else: # Interactive mode
            retrain_choice = input("Would you like to retrain the model with this new data now? (y/n): ").lower()
            if retrain_choice == 'y':
                logger.info("\n[*] Merging any other auto-labeled data first...")
                merge_auto_labeled_data()
                logger.info("[*] Retraining model...")
                train_model() # Call the actual training function
                logger.info("✅ Model retraining complete.")
            else:
                logger.info("[*] Model retraining skipped.")
    else:
        logger.info("\n[*] No new knowledge was curated in this session.")

    logger.info("--- 🧠 Knowledge Curation Session Finished ---")


if __name__ == '__main__':
    # You would typically call run_curation_session(autonomous_mode=True) from main.py
    run_curation_session(autonomous_mode=False)