import joblib
import os
from config import MODEL_PATH, AUTO_LABELED_CSV, CONFIDENCE_THRESHOLD

def load_pipeline(model_path=MODEL_PATH):
    """
    Loads the entire model pipeline (vectorizer + classifier) from a file.
    """
    # The .pkl file now contains the whole pipeline.
    pipeline = joblib.load(model_path)
    return pipeline

def predict_with_confidence(text):
    """
    Makes a prediction and gets the confidence score using the loaded pipeline.
    The pipeline handles both vectorizing the text and making the prediction.
    """
    pipeline = load_pipeline()

    # Get the probabilities for each class
    # The pipeline's predict_proba method takes the raw text directly.
    try:
        probs = pipeline.predict_proba([text])[0]
        classes = pipeline.classes_
    except AttributeError:
        # Some models like LinearSVC don't have predict_proba by default.
        # We can use decision_function as a proxy for confidence.
        decision_values = pipeline.decision_function([text])[0]
        # Simple normalization to get a confidence-like score
        probs = (decision_values - decision_values.min()) / (decision_values.max() - decision_values.min())
        classes = pipeline.classes_


    # Find the index of the highest probability/score
    best_idx = probs.argmax()

    # Return the class name and the confidence score
    return classes[best_idx], probs[best_idx]

def maybe_auto_label(text, label, confidence):
    """Saves text to a CSV for retraining if confidence is high."""
    if confidence >= CONFIDENCE_THRESHOLD:
        try:
            # Create file with header if it doesn't exist
            if not os.path.exists(AUTO_LABELED_CSV):
                with open(AUTO_LABELED_CSV, "w", encoding="utf-8") as f:
                    f.write("text,label\n")
            
            # Append new data
            with open(AUTO_LABELED_CSV, "a", encoding="utf-8") as f:
                clean_text = f"\"{text.replace('\"', '\"\"')}\"" # Handle quotes in CSV
                f.write(f"{clean_text},{label}\n")
            print(f"[📥] Auto-labeled and saved to {AUTO_LABELED_CSV}")
        except Exception as e:
            print(f"[!] Failed to auto-label: {e}")
    else:
        print(f"[✋] Confidence too low ({confidence:.2f}), not auto-labeling.")
