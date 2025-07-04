import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
from config import LABELED_DATA_CSV, MODEL_PATH, VECTORIZER_PATH
import warnings
from sklearn.exceptions import ConvergenceWarning

# Optional: Suppress warnings after fixing the causes
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def train_model():
    """
    Trains and evaluates a classification model by searching for the best
    model and parameters using GridSearchCV.
    """
    print(f"[*] Loading data from {LABELED_DATA_CSV}...")
    
    try:
        df = pd.read_csv(LABELED_DATA_CSV, engine='python', on_bad_lines='warn')
    except Exception as e:
        print(f"[!] Failed to read CSV file: {e}")
        return

    df.dropna(subset=['text', 'label'], inplace=True)
    df = df[df['text'].str.strip().astype(bool)]
    
    if len(df) < 10:
        print("[!] Not enough data to train a model. Need at least 10 examples.")
        return

    print(f"[*] Training with {len(df)} labeled examples.")

    label_counts = df['label'].value_counts()
    if (label_counts < 2).any():
        print("[!] Warning: One or more labels have only one example. Disabling stratification.")
        stratify_option = None
    else:
        stratify_option = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=stratify_option
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression())  # Placeholder, will be overwritten in GridSearchCV
    ])

    # --- Parameter Grid for GridSearchCV ---
    parameters = [
        {
            'tfidf__ngram_range': [(1, 2)],
            'tfidf__max_features': [7500, 10000],
            'tfidf__max_df': [0.75],
            'clf': [LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')],
            'clf__C': [1, 5, 10],
        },
        {
            'tfidf__ngram_range': [(1, 2)],
            'tfidf__max_features': [7500, 10000],
            'tfidf__max_df': [0.75],
            'clf': [LogisticRegression(solver='saga', class_weight='balanced', max_iter=5000, random_state=42)],
            'clf__C': [0.1, 1, 5],
        },
        {
            'tfidf__ngram_range': [(1, 2)],
            'tfidf__max_features': [7500, 10000],
            'tfidf__max_df': [0.75],
            'clf': [LinearSVC(random_state=42, dual=True, max_iter=3000, class_weight='balanced')],
            'clf__C': [0.1, 1, 5],
        }
    ]

    print("\n[*] Performing Grid Search to find the best model and parameters...")
    grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    print("\n--- Best Model Found ---")
    print(f"Best score: {grid_search.best_score_:.2f}")
    print("Best parameters set:")
    for param_name in sorted(grid_search.best_params_.keys()):
        print(f"\t{param_name}: {grid_search.best_params_[param_name]}")

    best_model = grid_search.best_estimator_

    print("\n--- Final Model Evaluation on Test Set ---")
    y_pred = best_model.predict(X_test)
    print(f"Accuracy on test set: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(best_model, MODEL_PATH)
    print(f"\n[✓] Best model pipeline saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
