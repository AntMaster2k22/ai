import time
import sys

# --- Project Imports ---
# This script assumes it's in the same directory as your other project files.
from config import URLS_FILE
from my_scraper import scrape_text_from_url
from embedder import embed_text
from memory import Memory
from model import predict_with_confidence, maybe_auto_label

def learn_from_file():
    """
    Reads a list of URLs from a file, classifies them using the trained model,
    and adds them to the AI's memory.
    """
    print("--- Starting Batch Learning from File ---")
    
    try:
        with open(URLS_FILE, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"[!] Error: The file '{URLS_FILE}' was not found.")
        return

    if not urls:
        print("[!] The URL file is empty. Nothing to learn.")
        return

    # Initialize memory once before the loop
    memory = Memory()
    total_urls = len(urls)
    print(f"[+] Found {total_urls} URLs to process.")

    for i, url in enumerate(urls):
        print(f"\n--- Processing URL {i+1}/{total_urls}: {url} ---")
        
        try:
            text = scrape_text_from_url(url)
            if not text:
                print("[!] No text could be scraped. Skipping.")
                continue

            # Get a prediction for the scraped text
            category, confidence = predict_with_confidence(text)
            print(f"[🧠] Prediction: {category} (Confidence: {confidence:.2f})")
            
            # Attempt to auto-label the data for future retraining
            maybe_auto_label(text[:1500], category, confidence)

            # Create the vector and metadata for memory
            vector = embed_text(text)
            metadata = {
                "url": url,
                "text_snippet": text[:500],
                "predicted_category": category,
                "confidence": round(confidence, 2)
            }
            
            # Add the new knowledge to memory
            memory.add(vector, metadata)
            print(f"[✓] Successfully added to memory.")

        except KeyboardInterrupt:
            print("\n\n[!] Batch learning interrupted by user. Exiting.")
            sys.exit(0)
        except Exception as e:
            print(f"[!] An unexpected error occurred with URL {url}: {e}")
            
        # Optional: add a small delay to be respectful to servers
        time.sleep(0.5) 

    print("\n--- Batch learning complete! ---")


if __name__ == "__main__":
    learn_from_file()
