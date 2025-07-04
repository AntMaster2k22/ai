import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from collections import Counter
import warnings

# --- Project Imports ---
# This script assumes it's in the root of your project directory
from config import LABELED_DATA_CSV
from model import load_pipeline # We use your existing function to load the model

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def label_unlabeled_data():
    """
    An intelligent assistant to label data in your labeled_data.csv file.
    It uses your existing model to predict known labels and then helps
    discover new potential labels through clustering.
    """
    print("--- 🤖 AI Labeling Assistant ---")

    # 1. Load data and identify unlabeled rows
    try:
        df = pd.read_csv(LABELED_DATA_CSV, engine='python', on_bad_lines='skip')
        # Fill any potential NaN values in 'label' column with a placeholder
        df['label'] = df['label'].fillna('UNLABELED')
    except FileNotFoundError:
        print(f"[!] Error: The file '{LABELED_DATA_CSV}' was not found.")
        return

    unlabeled_df = df[df['label'] == 'UNLABELED'].copy()

    if unlabeled_df.empty:
        print("✅ No unlabeled data found. Your dataset is complete!")
        return

    print(f"[*] Found {len(unlabeled_df)} unlabeled rows to process.")

    # 2. Load your pre-trained model pipeline
    try:
        pipeline = load_pipeline()
        print("[*] Successfully loaded your trained model.")
    except FileNotFoundError:
        print("[!] Error: model.pkl not found. Please train your model first using learn.py.")
        return
    except Exception as e:
        print(f"[!] An error occurred while loading the model: {e}")
        return
        
    # 3. Predict probabilities for existing labels
    unlabeled_texts = unlabeled_df['text'].astype(str).tolist()
    
    try:
        # Use the TF-IDF vectorizer part of your pipeline
        tfidf_vectorizer = pipeline.named_steps['tfidf']
        classifier = pipeline.named_steps['clf']
        
        X_unlabeled_tfidf = tfidf_vectorizer.transform(unlabeled_texts)
        probs = classifier.predict_proba(X_unlabeled_tfidf)
        
    except AttributeError:
        print("[!] Your model does not support predict_proba. Using decision_function instead.")
        # Fallback for models like LinearSVC
        decision_values = pipeline.decision_function(unlabeled_texts)
        # Normalize to get a confidence-like score
        probs = np.array([(v - v.min()) / (v.max() - v.min()) if (v.max() - v.min()) > 0 else np.zeros_like(v) for v in decision_values])


    # 4. Separate high- and low-confidence predictions
    high_confidence_indices = []
    low_confidence_indices = []
    
    # Use the confidence threshold from your config file
    from config import CONFIDENCE_THRESHOLD 
    
    for i, prob_vector in enumerate(probs):
        if np.max(prob_vector) >= CONFIDENCE_THRESHOLD:
            high_confidence_indices.append(i)
        else:
            low_confidence_indices.append(i)
            
    print(f"\n[*] Applying {len(pipeline.classes_)} existing labels...")
    
    # Assign high-confidence labels
    for i in high_confidence_indices:
        original_df_index = unlabeled_df.index[i]
        predicted_label_index = np.argmax(probs[i])
        predicted_label = pipeline.classes_[predicted_label_index]
        df.loc[original_df_index, 'label'] = predicted_label
        
    print(f"✅ Automatically assigned labels to {len(high_confidence_indices)} rows with high confidence (>={CONFIDENCE_THRESHOLD}).")
    
    # --- Part 2: Discovering New Labels ---
    if not low_confidence_indices:
        print("\n✅ No low-confidence items found. All unlabeled data has been processed!")
    else:
        print(f"\n[*] Found {len(low_confidence_indices)} rows that don't fit existing labels well.")
        print("[*] Grouping these items to discover potential new topics...")
        
        # Use the TF-IDF vectors we already created for the low-confidence items
        low_conf_vectors = X_unlabeled_tfidf[low_confidence_indices]
        
        # Determine a reasonable number of new clusters (topics)
        # A simple heuristic: one new topic for every 20-50 low-confidence items, but at least 2 and at most 20
        num_clusters = max(2, min(20, len(low_confidence_indices) // 30))
        
        print(f"[*] Attempting to find {num_clusters} new topics...")
        
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        unlabeled_df.loc[unlabeled_df.index[low_confidence_indices], 'cluster'] = kmeans.fit_predict(low_conf_vectors)
        
        print("\n--- New Topic Discovery ---")
        print("I've grouped the remaining items into clusters. Please provide a name for each new topic.")
        print("For each topic, I'll show you the most common words to help you decide.")

        # Get top terms per cluster from TF-IDF
        terms = tfidf_vectorizer.get_feature_names_out()

        for i in range(num_clusters):
            print(f"\n--- Analyzing New Topic Cluster #{i} ---")
            
            # Get texts in the current cluster
            cluster_texts = unlabeled_df[unlabeled_df['cluster'] == i]['text'].tolist()
            
            # Find the top 15 most frequent words in this cluster
            # To do this, we'll re-vectorize just these texts and sum up the TF-IDF scores
            cluster_tfidf = tfidf_vectorizer.transform(cluster_texts)
            word_scores = np.array(cluster_tfidf.sum(axis=0)).ravel()
            top_word_indices = word_scores.argsort()[-15:][::-1]
            top_words = [terms[j] for j in top_word_indices]
            
            print(f"💡 Top keywords for this topic: {', '.join(top_words)}")
            print(f"👀 Example text: '{cluster_texts[0][:200]}...'")

            new_label = input(f"Enter a new label name for Topic #{i} (or 'skip' to ignore): ").strip().lower().replace(" ", "_")
            
            if new_label and new_label != 'skip':
                # Assign this new label to all items in the cluster
                cluster_original_indices = unlabeled_df[unlabeled_df['cluster'] == i].index
                df.loc[cluster_original_indices, 'label'] = new_label
                print(f"✅ Assigned new label '{new_label}' to {len(cluster_original_indices)} items.")
            else:
                print(f"Skipping Topic #{i}.")

    # 7. Save the completed file
    output_filename = 'data/labeled_data_completed.csv'
    df.to_csv(output_filename, index=False)
    
    print(f"\n🎉 Success! Your new file '{output_filename}' has been created.")
    print("Final counts of all labels:")
    print(df['label'].value_counts())

if __name__ == "__main__":
    label_unlabeled_data()