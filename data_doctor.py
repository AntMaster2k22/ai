import pandas as pd
import os
import csv

# --- Project Imports ---
from config import LABELED_DATA_CSV

def run_data_checkup():
    """
    Scans the dataset for common issues like rare labels and offers
    interactive ways to fix them.
    """
    print("\n--- 🩺 AI Data Doctor ---")
    
    try:
        df = pd.read_csv(
            LABELED_DATA_CSV,
            engine='python',
            on_bad_lines='warn',
            quoting=csv.QUOTE_ALL,
            escapechar='\\'
        )
        df.dropna(subset=['text', 'label'], inplace=True)
    except FileNotFoundError:
        print(f"[!] The data file '{LABELED_DATA_CSV}' was not found.")
        return

    # --- Check 1: Find "Lonely" Labels (classes with very few examples) ---
    label_counts = df['label'].value_counts()
    # Let's define "lonely" as any label with fewer than 3 examples.
    lonely_labels = label_counts[label_counts < 3]

    if lonely_labels.empty:
        print("✅ Your dataset looks healthy! No labels with very few examples were found.")
        return

    print(f"[!] Found {len(lonely_labels)} labels with fewer than 3 examples. These can hurt model performance.")
    print("It's recommended to either find more examples for these labels or merge them into a more general category.")
    
    print("\n--- Lonely Labels ---")
    print(lonely_labels)
    
    fix_choice = input("\nWould you like to interactively fix these labels now? (y/n): ").lower()
    
    if fix_choice != 'y':
        print("[*] Okay. You can run this check-up again later.")
        return

    # Interactive fixing process
    for label_name, count in lonely_labels.items():
        print(f"\n--- Fixing '{label_name}' (has {count} example(s)) ---")
        
        # Get the rows for this label
        rows_to_fix = df[df['label'] == label_name]
        print("Example Text:")
        for index, row in rows_to_fix.iterrows():
            print(f"- {row['text'][:200]}...")

        action = input("Choose an action: [m]erge into another label, [d]elete these examples, or [s]kip: ").lower()

        if action == 'm':
            target_label = input(f"Which existing label should '{label_name}' be merged into? ").strip().lower()
            if target_label:
                # Find the indices to update and change the label
                indices_to_update = df[df['label'] == label_name].index
                df.loc[indices_to_update, 'label'] = target_label
                print(f"✅ Merged '{label_name}' into '{target_label}'.")
            else:
                print("[!] No target label provided. Skipping.")
        
        elif action == 'd':
            indices_to_delete = df[df['label'] == label_name].index
            df.drop(indices_to_delete, inplace=True)
            print(f"✅ Deleted {count} example(s) for label '{label_name}'.")
        
        else:
            print(f"[*] Skipping '{label_name}'.")

    # Save the cleaned dataframe
    save_choice = input("\nSave all changes back to the main data file? (y/n): ").lower()
    if save_choice == 'y':
        df.to_csv(LABELED_DATA_CSV, index=False, quoting=csv.QUOTE_ALL, escapechar='\\')
        print(f"✅ All changes have been saved to '{LABELED_DATA_CSV}'.")
        print("[*] It's highly recommended to retrain your model now.")
    else:
        print("[*] Changes discarded.")


if __name__ == "__main__":
    run_data_checkup()