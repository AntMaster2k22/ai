import faiss
import numpy as np
import os
import pickle
from config import VECTOR_DIMENSION, INDEX_PATH, METADATA_PATH

class Memory:
    """Manages the FAISS vector index and metadata for the AI."""
    def __init__(self, dim=VECTOR_DIMENSION, index_path=INDEX_PATH, meta_path=METADATA_PATH):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path

        if os.path.exists(self.index_path):
            print("[*] Loading existing memory index from disk.")
            self.index = faiss.read_index(self.index_path)
        else:
            print("[*] No existing index found, creating a new one.")
            self.index = faiss.IndexFlatL2(self.dim)

        if os.path.exists(self.meta_path):
            with open(self.meta_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            self.metadata = []

    def add(self, vector: np.ndarray, meta: dict):
        """Adds a vector and its metadata to the in-memory index."""
        if not isinstance(vector, np.ndarray):
            raise TypeError("Vector must be a numpy array.")
        # FAISS expects a 2D array, so we reshape the 1D vector
        self.index.add(np.array([vector], dtype=np.float32))
        self.metadata.append(meta)

    def query(self, vector: np.ndarray, k: int = 3):
        """Queries the index for the top k most similar vectors."""
        if self.index.ntotal == 0:
            return []
        distances, indices = self.index.search(np.array([vector], dtype=np.float32), k)
        
        # Filter out invalid -1 indices that FAISS can return if k > ntotal
        results = [self.metadata[i] for i in indices[0] if i != -1]
        return results

    def save(self):
        """Saves the current state of the index and metadata to disk."""
        print(f"[*] Writing FAISS index to {self.index_path}...")
        faiss.write_index(self.index, self.index_path)
        
        print(f"[*] Writing metadata to {self.meta_path}...")
        with open(self.meta_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        print("[✓] Memory state saved successfully.")