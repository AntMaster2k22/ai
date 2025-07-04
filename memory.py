# memory.py

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

        # Load index if it exists, otherwise create a new one
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            print("[+] Loaded existing memory index.")
        else:
            self.index = faiss.IndexFlatL2(self.dim)
            print("[+] No existing index found, creating a new one.")

        # Load metadata if it exists
        if os.path.exists(self.meta_path):
            with open(self.meta_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            self.metadata = []

    def add(self, vector: np.ndarray, meta: dict):
        """Adds a vector and its metadata to memory and saves."""
        if not isinstance(vector, np.ndarray):
            raise TypeError("Vector must be a numpy array.")
        self.index.add(np.array([vector], dtype=np.float32))
        self.metadata.append(meta)
        self.save()

    def query(self, vector: np.ndarray, k: int = 3):
        """Queries the index for the top k most similar vectors."""
        if self.index.ntotal == 0:
            return []
        distances, indices = self.index.search(np.array([vector], dtype=np.float32), k)
        # Filter out invalid indices that can sometimes occur
        results = [self.metadata[i] for i in indices[0] if i != -1]
        return results

    def save(self):
        """Saves the index and metadata to disk."""
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, 'wb') as f:
            pickle.dump(self.metadata, f)