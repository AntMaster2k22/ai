import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import joblib
import time
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning

from config import LABELED_DATA_CSV, MODEL_PATH

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def train_model():
    """
    Trains a model and provides an upfront time estimate before starting
    the full grid search.
    """
    print(f"[*] Loading data from {LABELED_DATA_CSV}...")
    
    try:
        df = pd.read_csv(LABELED_DATA_CSV, engine='pyarrow')
    except Exception:
        print("[!] Pyarrow engine failed, falling back to python engine.")
        df = pd.read_csv(LABELED_DATA_CSV, engine='python', on_bad_lines='warn')

    df.dropna(subset=['text', 'label'], inplace=True)
    df = df[df['text'].str.strip().astype(bool)]
    
    if len(df) < 10:
        print("[!] Not enough data to train a model. Need at least 10 examples.")
        return

    print(f"[*] Training with {len(df)} labeled examples.")

    stratify_option = None
    if df['label'].nunique() > 1 and all(df['label'].value_counts() >= 2):
        stratify_option = df["label"]
    else:
        print("[!] Warning: One or more labels have insufficient samples for stratification.")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=stratify_option
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression()) # Placeholder
    ])

    parameters = [
        {
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'tfidf__max_features': [8000],
            'clf': [LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')],
            'clf__C': [1, 10],
        },
        {
            'tfidf__ngram_range': [(1, 2)],
            'tfidf__max_features': [8000],
            'clf': [LinearSVC(random_state=42, dual=True, max_iter=3000, class_weight='balanced')],
            'clf__C': [0.1, 1],
        }
    ]
    
    # --- 1. ESTIMATE THE TRAINING TIME ---
    print("\n[*] Calculating estimated training time...")
    try:
        # Create a grid of all parameter combinations
        param_grid = list(ParameterGrid(parameters))
        total_fits = len(param_grid) * 3 # 3 is the 'cv' value
        
        # Time a single fit on a small subset of the data for a quick estimate
        sample_X = X_train[:500]
        sample_y = y_train[:500]
        
        # Use the first parameter set for the test run
        p = param_grid[0]
        pipeline.set_params(**p)
        
        start_est = time.time()
        pipeline.fit(sample_X, sample_y)
        end_est = time.time()
        
        # Scale the estimate to the full dataset size
        time_for_one_fit = (end_est - start_est) * (len(X_train) / len(sample_X))
        estimated_total_time = time_for_one_fit * total_fits
        
        print(f"[*] This process will perform {total_fits} training fits.")
        print(f"[*] ROUGH ESTIMATED TRAINING TIME: {estimated_total_time / 60:.1f} minutes.")
        input("    Press Enter to begin the training...")
        
    except Exception as e:
        print(f"[!] Could not calculate estimate, proceeding with training. Error: {e}")


    # --- 2. RUN THE ACTUAL TRAINING ---
    grid_search = GridSearchCV(pipeline, parameters, cv=3, n_jobs=-1, verbose=1)

    print("\n[*] Performing Grid Search to find the best model...")
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    end_time = time.time()
    
    actual_training_time = end_time - start_time
    print(f"\n--- ACTUAL TRAINING TIME: {actual_training_time / 60:.2f} minutes ---")

    print("\n--- Best Model Found ---")
    print(f"Best cross-validation score: {grid_search.best_score_:.2f}")
    print("Best parameters set:")
    for param_name in sorted(grid_search.best_params_.keys()):
        print(f"\t{param_name}: {grid_search.best_params_[param_name]}")

    best_model = grid_search.best_estimator_

    print("\n--- Final Model Evaluation on Test Set ---")
    y_pred = best_model.predict(X_test)
    from sklearn.metrics import accuracy_score, classification_report
    print(f"Accuracy on test set: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(best_model, MODEL_PATH)
    print(f"\n[✓] Best model pipeline saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()