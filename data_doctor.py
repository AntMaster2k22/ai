import pandas as pd
import os
import csv
import logging # Import logging
from io import StringIO # Needed for reading cleaned data back into pandas

# --- Project Imports ---
# Assuming HARVESTER_MIN_CONTENT_WORDS and LABELED_DATA_CSV are in config.py
from config import LABELED_DATA_CSV, HARVESTER_MIN_CONTENT_WORDS

# Configure logging (can be moved to config.py for global setup)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global list to store problematic line numbers
problematic_line_numbers = []

def bad_line_handler(line):
    """
    Custom handler for on_bad_lines.
    It logs the warning and stores the line number.
    """
    global problematic_line_numbers
    # The 'line' argument here is the raw text of the problematic line.
    # We'll use a different strategy to get the line number directly from the read operation.
    # For now, let's keep track of a placeholder and identify actual line numbers later
    # by comparing original and read content, or by pre-reading.

    # Simpler approach: when on_bad_lines='warn' is used, pandas already prints the line number.
    # We need to capture those lines before read_csv discards them.
    # The best way to implement 'delete all problematic lines' is to read the file line by line
    # and filter out bad lines *before* pandas processes it.
    
    # We'll re-architect this slightly: first, identify bad lines. Second, load good lines.
    # The current on_bad_lines='warn' doesn't directly give us the line content *to remove*
    # after the fact, as pandas has already skipped it.
    # So, we'll read the file raw, identify good/bad, then let pandas read the good ones.
    pass # This function won't be used directly with the new approach for 'c' option.


def run_data_checkup(autonomous_mode: bool = False):
    """
    Scans the dataset for common issues like rare labels, duplicate texts,
    and empty/short texts. Offers interactive ways to fix them, or
    applies default actions in autonomous mode.

    Args:
        autonomous_mode (bool): If True, the script runs non-interactively
                                and applies predefined clean-up rules.
    """
    logger.info("\n--- 🩺 AI Data Doctor Starting ---")

    global problematic_line_numbers
    problematic_line_numbers = [] # Reset for each run

    original_lines = []
    cleaned_temp_lines = []
    header = None

    # --- Step 1: Pre-scan the CSV to identify and filter problematic lines ---
    logger.info(f"[*] Pre-scanning '{LABELED_DATA_CSV}' for parsing issues...")
    try:
        with open(LABELED_DATA_CSV, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f, quoting=csv.QUOTE_ALL, escapechar='\\')
            
            # Read header
            try:
                header = next(reader)
                cleaned_temp_lines.append(','.join(f'"{h.replace("\"", "\"\"")}"' for h in header)) # Quote header correctly for re-writing
                original_lines.append(','.join(header)) # Store original header for comparison
            except StopIteration:
                logger.error(f"[x] The file '{LABELED_DATA_CSV}' is empty or has no header.")
                return pd.DataFrame() # Return empty DataFrame

            for i, row in enumerate(reader):
                line_num = i + 2 # +1 for 0-index, +1 for header
                original_row_str = ','.join(f'"{field.replace("\"", "\"\"")}"' for field in row) # Attempt to reconstruct original row string
                
                # Basic check for expected number of fields. Adjust '2' if your CSV has more columns.
                # This is the primary cause of your "Expected X fields, saw Y" warnings.
                if len(row) == 2: # Assuming your CSV is expected to have exactly 2 fields
                    cleaned_temp_lines.append(original_row_str)
                    original_lines.append(original_row_str)
                else:
                    problematic_line_numbers.append(line_num)
                    logger.warning(f"[*] Problematic line detected (Line {line_num}): Expected 2 fields, saw {len(row)}. Skipping this line for initial load.")
                    # We still store the original line to potentially show it to the user later if needed,
                    # but it won't be part of the 'cleaned_temp_lines' passed to pandas initially.
                    original_lines.append(original_row_str)

    except FileNotFoundError:
        logger.error(f"[!] The data file '{LABELED_DATA_CSV}' was not found. Please ensure it exists.")
        return
    except pd.errors.EmptyDataError:
        logger.warning(f"[!] The data file '{LABELED_DATA_CSV}' is empty. No data to check.")
        return
    except Exception as e:
        logger.error(f"[x] An unexpected error occurred while pre-scanning data: {e}")
        return

    df = pd.DataFrame() # Initialize df outside try block

    if problematic_line_numbers and not autonomous_mode:
        print("\n--- ❗ Data Parsing Issues Detected ❗ ---")
        print(f"Found {len(problematic_line_numbers)} lines with parsing errors (e.g., incorrect number of fields).")
        print(f"Problematic lines: {', '.join(map(str, problematic_line_numbers[:10]))}{'...' if len(problematic_line_numbers) > 10 else ''}")

        action = input("Choose an action: [y] continue (skip problematic lines), [n] exit, [c] delete ALL problematic lines from the file: ").lower()

        if action == 'c':
            try:
                # Reconstruct the DataFrame only with good lines
                # The header is already the first element in cleaned_temp_lines
                string_data = StringIO('\n'.join(cleaned_temp_lines))
                df = pd.read_csv(string_data, quoting=csv.QUOTE_ALL, escapechar='\\')
                
                # Overwrite the original file with only the good lines (including header)
                with open(LABELED_DATA_CSV, 'w', encoding='utf-8', newline='') as f:
                    f.write('\n'.join(cleaned_temp_lines))
                logger.info(f"✅ Successfully deleted all {len(problematic_line_numbers)} problematic lines from '{LABELED_DATA_CSV}'.")
                problematic_line_numbers = [] # Clear the list as they are now deleted

            except Exception as e:
                logger.error(f"[x] Error deleting problematic lines and reloading data: {e}")
                logger.warning("[!] Data state might be inconsistent. Please check your CSV manually.")
                return # Exit if critical error during deletion/reload
        elif action == 'n':
            logger.info("[*] Exiting without making any changes due to parsing issues.")
            return # Exit if user chooses 'n'
        else: # action == 'y' or any other input, proceed by skipping
            logger.info("[*] Proceeding by skipping problematic lines for this session.")
            # Load the data, knowing pandas will warn/skip these lines
            # For consistency, we should load from the `cleaned_temp_lines` if 'c' wasn't chosen,
            # because the pre-scan already identified them.
            string_data = StringIO('\n'.join(cleaned_temp_lines))
            df = pd.read_csv(string_data, quoting=csv.QUOTE_ALL, escapechar='\\')

    else: # No problematic lines or autonomous mode
        logger.info("[*] No parsing issues detected or running in autonomous mode. Loading data...")
        try:
            # If no parsing issues were found in pre-scan, or in autonomous mode,
            # we can directly load the original file with 'on_bad_lines=warn'
            # (although our pre-scan already handled the 'skip' part for 'c' option)
            # For simplicity after the pre-scan, we'll always load from the prepared string data
            string_data = StringIO('\n'.join(cleaned_temp_lines))
            df = pd.read_csv(string_data, quoting=csv.QUOTE_ALL, escapechar='\\')
        except Exception as e:
            logger.error(f"[x] Error loading data after pre-scan: {e}")
            return


    # Initial cleanup for missing text/label, applied regardless of parsing errors
    initial_rows_after_load = len(df)
    df.dropna(subset=['text', 'label'], inplace=True)
    if len(df) < initial_rows_after_load:
        logger.info(f"Removed {initial_rows_after_load - len(df)} rows with missing 'text' or 'label' after initial load.")

    if df.empty:
        logger.warning("[!] Dataset is empty after initial loading and cleanup. Nothing more to check.")
        return

    # --- Check 1: Find "Lonely" Labels (classes with very few examples) ---
    label_counts = df['label'].value_counts()
    lonely_labels = label_counts[label_counts < 3] # Let's define "lonely" as any label with fewer than 3 examples.

    if not lonely_labels.empty:
        logger.warning(f"[!] Found {len(lonely_labels)} labels with fewer than 3 examples. These can hurt model performance.")
        logger.info("It's recommended to either find more examples for these labels or merge them into a more general category.")
        logger.info("\n--- Lonely Labels ---")
        logger.info(label_counts.to_string()) # Use to_string() for better logging output

        if not autonomous_mode:
            fix_choice = input("\nWould you like to interactively fix these labels now? (y/n): ").lower()
            if fix_choice == 'y':
                for label_name, count in lonely_labels.items():
                    logger.info(f"\n--- Fixing '{label_name}' (has {count} example(s)) ---")
                    rows_to_fix = df[df['label'] == label_name]
                    logger.info("Example Text (first 200 chars):")
                    for _, row in rows_to_fix.iterrows():
                        logger.info(f"- {row['text'][:200]}...")

                    action = input("Choose an action: [m]erge into another label, [d]elete these examples, or [s]kip: ").lower()

                    if action == 'm':
                        target_label = input(f"Which existing label should '{label_name}' be merged into? ").strip().lower()
                        if target_label:
                            indices_to_update = df[df['label'] == label_name].index
                            df.loc[indices_to_update, 'label'] = target_label
                            logger.info(f"✅ Merged '{label_name}' into '{target_label}'.")
                        else:
                            logger.info("[!] No target label provided. Skipping merge.")

                    elif action == 'd':
                        indices_to_delete = df[df['label'] == label_name].index
                        df.drop(indices_to_delete, inplace=True)
                        logger.info(f"✅ Deleted {count} example(s) for label '{label_name}'.")

                    else:
                        logger.info(f"[*] Skipping '{label_name}'.")
            else:
                logger.info("[*] Lonely label fixing skipped for now.")
        else:
            logger.info("[*] Running in autonomous mode: Lonely labels detected but no automatic action configured. Manual review recommended.")
    else:
        logger.info("✅ Dataset looks healthy! No labels with very few examples were found.")

    # --- Check 2: Find Duplicate Text Entries ---
    initial_rows = len(df)
    df.drop_duplicates(subset=['text'], inplace=True)
    if len(df) < initial_rows:
        removed_duplicates = initial_rows - len(df)
        logger.info(f"[*] Removed {removed_duplicates} duplicate text entries.")
        if not autonomous_mode:
            print(f"Removed {removed_duplicates} duplicate text entries.") # Inform user in interactive mode

    # --- Check 3: Find Empty or Very Short Texts ---
    initial_rows = len(df)
    min_words_for_content = getattr(HARVESTER_MIN_CONTENT_WORDS, 'HARVESTER_MIN_CONTENT_WORDS', 10) # Corrected: should be config.HARVESTER_MIN_CONTENT_WORDS, not logging
    
    # Filter out rows where 'text' is not a string or is empty/too short
    df_cleaned = df[df['text'].apply(lambda x: isinstance(x, str) and len(x.split()) >= min_words_for_content)]
    
    removed_short_texts = initial_rows - len(df_cleaned)
    if removed_short_texts > 0:
        logger.info(f"[*] Removed {removed_short_texts} entries with empty or very short text (less than {min_words_for_content} words).")
        if not autonomous_mode:
            print(f"Removed {removed_short_texts} entries with empty or very short text.") # Inform user in interactive mode
        df = df_cleaned # Update df to the cleaned version

    if not autonomous_mode:
        # Save the cleaned dataframe - only prompt in interactive mode
        save_choice = input("\nSave all changes back to the main data file? (y/n): ").lower()
        if save_choice == 'y':
            try:
                df.to_csv(LABELED_DATA_CSV, index=False, quoting=csv.QUOTE_ALL, escapechar='\\')
                logger.info(f"✅ All changes have been saved to '{LABELED_DATA_CSV}'.")
                logger.info("[*] It's highly recommended to retrain your model now.")
            except IOError as e:
                logger.error(f"[x] Error saving changes to '{LABELED_DATA_CSV}': {e}")
        else:
            logger.info("[*] Changes discarded for this session.")
    else:
        # In autonomous mode, always save changes
        try:
            df.to_csv(LABELED_DATA_CSV, index=False, quoting=csv.QUOTE_ALL, escapechar='\\')
            logger.info(f"✅ Changes applied and saved to '{LABELED_DATA_CSV}' in autonomous mode.")
            logger.info("[*] Model re-training should be considered after autonomous data clean-up.")
        except IOError as e:
            logger.error(f"[x] Error saving changes to '{LABELED_DATA_CSV}' in autonomous mode: {e}")

    logger.info("--- 🩺 AI Data Doctor Finished ---")


if __name__ == "__main__":
    run_data_checkup(autonomous_mode=False) # Default to interactive for direct execution