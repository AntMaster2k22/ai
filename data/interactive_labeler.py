import pandas as pd
import sys
import os

# --- Project Imports ---
# This script assumes it's in the same directory as your other project files.
from config import URLS_FILE, LABELED_DATA_CSV
from my_scraper import scrape_text_from_url
from model import predict_with_confidence

class InteractiveLabeler:
    """
    A robust class to handle the interactive labeling of URLs,
    with features like undo, skip, and progress tracking.
    """

    def __init__(self, urls_file, labeled_data_csv):
        self.urls_file = urls_file
        self.labeled_data_csv = labeled_data_csv
        self.urls_to_process = self._load_urls()
        self.existing_texts = self._load_existing_data()
        self.session_labeled_count = 0
        self.last_snippet_added = None

    def _load_urls(self):
        """Loads URLs from the specified file."""
        try:
            with open(self.urls_file, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"[!] Error: The file '{self.urls_file}' was not found.")
            return []

    def _load_existing_data(self):
        """Loads existing labeled data to avoid re-labeling."""
        if not os.path.exists(self.labeled_data_csv):
            with open(self.labeled_data_csv, 'w', encoding='utf-8') as f:
                f.write("text,label\n")
            return set()
        try:
            df = pd.read_csv(self.labeled_data_csv, engine='python')
            return set(df['text'])
        except pd.errors.EmptyDataError:
             return set()


    def _save_label(self, snippet, label):
        """Saves a new label to the CSV file."""
        with open(self.labeled_data_csv, 'a', encoding='utf-8') as f:
            # Ensure proper CSV quoting
            f.write(f'"{snippet}",{label}\n')
        self.existing_texts.add(snippet)
        self.last_snippet_added = snippet # Remember for undo
        self.session_labeled_count += 1

    def _undo_last_label(self):
        """Removes the last saved label from the CSV file."""
        if not self.last_snippet_added:
            print("[!] No action in this session to undo.")
            return

        try:
            df = pd.read_csv(self.labeled_data_csv, engine='python')
            # Check if the last added text is actually the last line
            if not df.empty and df.iloc[-1]['text'] == self.last_snippet_added:
                df = df.iloc[:-1] # Remove the last row
                df.to_csv(self.labeled_data_csv, index=False)
                self.existing_texts.remove(self.last_snippet_added)
                self.session_labeled_count -= 1
                self.last_snippet_added = None # Clear undo state
                print("[✓] Last label has been successfully undone.")
            else:
                print("[!] Undo failed. The last line in the file does not match the last added label.")

        except (FileNotFoundError, pd.errors.EmptyDataError):
            print("[!] Error: Could not read the labeled data file to undo.")


    def run(self, num_to_label=20):
        """Runs the main interactive labeling session."""
        print("--- Interactive Labeling Session ---")
        print("Provide the correct label for the snippet shown.")
        print("Commands: type 'skip', 'quit', or 'undo'.")

        for url in self.urls_to_process:
            if self.session_labeled_count >= num_to_label:
                break

            print(f"\n--- Processing URL: {url} ---")

            try:
                text = scrape_text_from_url(url)
                if not text:
                    print("[!] Could not scrape text. Skipping.")
                    continue
            except Exception as e:
                print(f"[!] Error scraping {url}: {e}. Skipping.")
                continue

            snippet = text[:1500].replace('"', '""')
            if snippet in self.existing_texts:
                print("[!] This content has already been labeled. Skipping.")
                continue

            try:
                prediction, confidence = predict_with_confidence(text)
                print(f"[🧠] Model's best guess: '{prediction}' (Confidence: {confidence:.2f})")
            except Exception:
                prediction = ""

            print(f"\nSnippet:\n---\n{snippet[:500]}...\n---")

            while True: # Loop to handle input until a valid action is taken
                try:
                    user_input = input(f"Enter label (or press Enter for '{prediction}'): ").lower().strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n\n--- Exiting labeling session. ---")
                    sys.exit(0)

                if user_input == 'quit':
                    print(f"\n--- Session complete. You added {self.session_labeled_count} new labels. ---")
                    return
                elif user_input == 'skip':
                    break
                elif user_input == 'undo':
                    self._undo_last_label()
                    # After undoing, we still stay on the same URL to give the user a chance to re-label it
                    print(f"\nRe-processing URL: {url}")
                    print(f"\nSnippet:\n---\n{snippet[:500]}...\n---")
                    continue
                else:
                    final_label = user_input if user_input else prediction
                    if final_label:
                        self._save_label(snippet, final_label)
                        print(f"[✓] Saved label '{final_label}'. ({self.session_labeled_count}/{num_to_label})")
                        break # Move to the next URL
                    else:
                        print("[!] No label provided. Please enter a label or a command.")

if __name__ == "__main__":
    labeler = InteractiveLabeler(URLS_FILE, LABELED_DATA_CSV)
    labeler.run()
