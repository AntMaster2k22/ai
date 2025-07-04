import os
import sys
from my_scraper import scrape_text_from_url
from embedder import embed_text
from memory import Memory
from model import predict_with_confidence, maybe_auto_label
from config import AUTO_LABELED_CSV, CONFIDENCE_THRESHOLD

def learn_from_url():
    """Scrapes a URL, classifies its content, and adds it to memory."""
    memory = Memory()
    url = input("Enter a URL to learn from: ")
    if not url:
        return

    print(f"\n[+] Scraping text from {url}...")
    text = scrape_text_from_url(url)
    if not text:
        print("[!] No text could be scraped. Aborting.")
        return

    print("[+] Scraped text. Embedding...")
    vector = embed_text(text)
    
    print("[+] Embedded. Classifying...")
    try:
        category, confidence = predict_with_confidence(text)
        print(f"[🧠] Model prediction: {category} (confidence: {confidence:.2f})")
        maybe_auto_label(text[:1000], category, confidence)
    except Exception as e:
        print(f"[!] Model prediction failed: {e}")
        category = "unknown"
        confidence = 0.0

    print("[+] Saving to memory...")
    metadata = {
        "url": url,
        "text_snippet": text[:500],
        "predicted_category": category,
        "confidence": round(confidence, 2)
    }
    memory.add(vector, metadata)
    print("[✓] Knowledge added to memory.")

def chat():
    """
    Initiates a chat session. If memory is populated, it queries the memory.
    If memory is empty, it uses the classification model directly on the query.
    """
    memory = Memory()
    has_memory = memory.index.ntotal > 0

    if has_memory:
        print("\n--- Chat with your Knowledge Base --- (type 'exit' to return to menu)")
    else:
        print("\n--- Chat with your Classifier --- (Memory is empty)")
        print("I can classify your text. Ask me anything! (type 'exit' to return to menu)")

    while True:
        try:
            query = input("🧠 You: ")
        except (EOFError, KeyboardInterrupt):
            print("\n\n--- Exiting chat. ---")
            break

        if query.lower() in ["exit", "quit"]:
            break
        if not query.strip():
            continue

        if has_memory:
            # Query the existing knowledge base
            q_vector = embed_text(query)
            results = memory.query(q_vector, k=1)

            if not results:
                print("⚠️ Nothing found in memory that matches your query.")
                continue

            best = results[0]
            print(f"\n🔎 Top Result from Memory:")
            print(f"🌐 URL: {best.get('url', 'N/A')}")
            print(f"📁 Topic: {best.get('predicted_category', 'unknown')} (Confidence: {best.get('confidence', 'N/A')})")
            print(f"💬 Snippet: {best.get('text_snippet', '')}...\n")
        else:
            # Use the classifier directly on the user's query
            try:
                category, confidence = predict_with_confidence(query)
                print(f"\n[CLASSIFICATION]")
                print(f"💬 I think your query is about: '{category}'")
                print(f"Confidence: {confidence:.2f}\n")
            except Exception as e:
                print(f"[!] Could not classify your query: {e}")


def main_menu():
    """Displays the main menu and handles user choices."""
    while True:
        print("\n--- Local Semantic AI ---")
        print("1. Learn from a new URL")
        print("2. Chat with your AI")
        print("3. Exit")
        
        try:
            choice = input("> ")
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if choice == '1':
            learn_from_url()
        elif choice == '2':
            chat()
        elif choice == '3':
            print("Goodbye!")
            break
        else:
            print("[!] Invalid choice, please try again.")

if __name__ == "__main__":
    main_menu()
