import pandas as pd
import os
import csv

# --- Project Imports ---
# This assumes the script is in the root folder, so config.py is accessible
from config import LABELED_DATA_CSV, AUTO_LABELED_CSV

def merge_files():
    """
    Safely merges the auto-labeled data into the main labeled_data.csv,
    using robust methods to handle complex text data.
    """
    print("[*] Starting merge process...")

    if not os.path.exists(AUTO_LABELED_CSV):
        print("[*] No auto_labeled_data.csv file found to merge. Exiting.")
        return

    try:
        # Use the robust CSV reading method to handle messy text
        print(f"[*] Reading main dataset from {LABELED_DATA_CSV}...")
        labeled_df = pd.read_csv(
            LABELED_DATA_CSV, 
            engine='python', 
            on_bad_lines='warn', 
            quoting=csv.QUOTE_ALL, 
            escapechar='\\'
        )

        print(f"[*] Reading new data from {AUTO_LABELED_CSV}...")
        auto_df = pd.read_csv(
            AUTO_LABELED_CSV, 
            engine='python', 
            on_bad_lines='warn', 
            quoting=csv.QUOTE_ALL, 
            escapechar='\\'
        )

        if auto_df.empty:
            print("[*] auto_labeled_data.csv is empty. Cleaning up.")
            os.remove(AUTO_LABELED_CSV)
            return

        # Combine, remove duplicates, and save
        print("[*] Combining datasets and removing duplicates...")
        combined_df = pd.concat([labeled_df, auto_df]).drop_duplicates(subset="text", keep='last')
        
        # Use the robust CSV writing method
        combined_df.to_csv(
            LABELED_DATA_CSV, 
            index=False, 
            quoting=csv.QUOTE_ALL, 
            escapechar='\\'
        )

        print(f"✅ Successfully merged {len(auto_df)} new entries into the main dataset.")
        
        # Clean up the auto-labeled file after successful merge
        os.remove(AUTO_LABELED_CSV)
        print(f"[*] Removed temporary file: {AUTO_LABELED_CSV}")

    except FileNotFoundError as e:
        print(f"[!] Error: A required file was not found. {e}")
    except Exception as e:
        print(f"[!] An unexpected error occurred during the merge: {e}")

if __name__ == "__main__":
    merge_files()