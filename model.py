import joblib
import os
from config import MODEL_PATH, AUTO_LABELED_CSV, CONFIDENCE_THRESHOLD

class Classifier:
    """A singleton-like class to hold and manage the loaded model pipeline."""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Classifier, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path=MODEL_PATH):
        if self._initialized:
            return
        self.pipeline = None
        if os.path.exists(model_path):
            print("[*] Loading classification model pipeline into memory...")
            self.pipeline = joblib.load(model_path)
            print("[+] Classification model loaded.")
        else:
            print("[!] Model pipeline not found. Please train a model first.")
        self._initialized = True

    def predict(self, text):
        """Makes a prediction and gets the confidence score."""
        if not self.pipeline:
            return "unclassified", 0.0

        try:
            probs = self.pipeline.predict_proba([text])[0]
            classes = self.pipeline.classes_
        except AttributeError:
            decision_values = self.pipeline.decision_function([text])[0]
            # Normalize to a 0-1 range for a confidence-like score
            if len(decision_values.shape) == 1 and decision_values.size > 1:
                 probs = (decision_values - decision_values.min()) / (decision_values.max() - decision_values.min())
            else: # Handle single class case
                 probs = np.array([1.0])
            classes = self.pipeline.classes_

        best_idx = probs.argmax()
        return classes[best_idx], probs[best_idx]

# --- Global instance ---
# This line creates the model object that the rest of your app will use.
# It's only loaded from disk the first time it's imported.
model_pipeline = Classifier()

def predict_with_confidence(text):
    """Public function to access the loaded model."""
    return model_pipeline.predict(text)

def maybe_auto_label(text, label, confidence):
    """Saves text to a CSV for retraining if confidence is high."""
    if confidence >= CONFIDENCE_THRESHOLD:
        try:
            file_exists = os.path.exists(AUTO_LABELED_CSV)
            with open(AUTO_LABELED_CSV, "a", encoding="utf-8", newline='') as f:
                if not file_exists:
                    f.write("text,label\n")
                
                # Proper CSV formatting for text with quotes/commas
                import csv
                writer = csv.writer(f)
                writer.writerow([text, label])

            print(f"[📥] Auto-labeled and saved to {AUTO_LABELED_CSV}")
        except Exception as e:
            print(f"[!] Failed to auto-label: {e}")
    else:
        print(f"[✋] Confidence too low ({confidence:.2f}), not auto-labeling.")