import time
import sys
import numpy as np

# --- Project Imports ---
from config import URLS_FILE
from my_scraper import scrape_text_from_url
from embedder import embed_text
from memory import Memory
from model import predict_with_confidence, maybe_auto_label

def learn_from_file():
    """
    Reads a list of URLs, classifies them, and adds them to memory.
    Optimized for batch performance and provides time estimates.
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

    memory = Memory()
    total_urls = len(urls)
    time_per_url = []
    
    print(f"[+] Found {total_urls} URLs to process.")

    for i, url in enumerate(urls):
        iter_start_time = time.time()
        print(f"\n--- Processing URL {i+1}/{total_urls}: {url} ---")
        
        try:
            text = scrape_text_from_url(url)
            if not text:
                print("[!] No text could be scraped. Skipping.")
                continue

            category, confidence = predict_with_confidence(text)
            print(f"[🧠] Prediction: {category} (Confidence: {confidence:.2f})")
            
            maybe_auto_label(text[:1500], category, confidence)

            vector = embed_text(text)
            metadata = {
                "url": url,
                "text_snippet": text[:500],
                "predicted_category": category,
                "confidence": round(confidence, 2)
            }
            
            # Add to memory without saving to disk immediately
            memory.add(vector, metadata)
            print(f"[+] Added to memory object.")

        except KeyboardInterrupt:
            print("\n\n[!] Batch learning interrupted by user.")
            break
        except Exception as e:
            print(f"[!] An unexpected error occurred with URL {url}: {e}")

        # --- Time Estimation Logic ---
        iter_time = time.time() - iter_start_time
        time_per_url.append(iter_time)
        avg_time = np.mean(time_per_url)
        remaining_urls = total_urls - (i + 1)
        estimated_remaining_time = avg_time * remaining_urls
        print(f"[*] Time for this URL: {iter_time:.2f}s. Approx. time remaining: {estimated_remaining_time / 60:.2f} minutes.")
            
        time.sleep(0.5)

    # --- Save memory once at the end ---
    print("\n[*] Saving all new entries to memory index...")
    memory.save()
    print("[✓] Memory saved successfully.")
    
    print("\n--- Batch learning complete! ---")

if __name__ == "__main__":
    learn_from_file()