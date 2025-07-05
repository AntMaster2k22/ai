from sentence_transformers import SentenceTransformer
import numpy as np

class TextEmbedder:
    """Singleton class to manage the SentenceTransformer model."""
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TextEmbedder, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        if self._initialized:
            return
        print(f"[*] Initializing the sentence embedding model ('{model_name}')...")
        print("[!] This may take a moment the first time it's run.")
        self.model = SentenceTransformer(model_name)
        print("[+] Embedding model loaded into memory.")
        self._initialized = True

    def embed(self, text: str) -> np.ndarray:
        """Encodes a single string of text into a vector."""
        return self.model.encode(text)

# --- Global instance ---
# This object is created once and reused across the application.
embedder_instance = TextEmbedder()

def embed_text(text: str) -> np.ndarray:
    """Public function to access the loaded embedding model."""
    return embedder_instance.embed(text)