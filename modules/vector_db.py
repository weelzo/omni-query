import faiss
import numpy as np

class VectorDB:
    def __init__(self, dim):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)

    def add(self, embeddings):
        # Convert to numpy array if not already
        embeddings_np = np.asarray(embeddings, dtype=np.float32)
        self.index.add(embeddings_np)

    def search(self, query_embedding, k=5):
        # Convert to numpy array and ensure correct shape
        query_np = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        distances, indices = self.index.search(query_np, k)
        return indices[0]
    
    def reset(self):
        """Reset the index, clearing all stored vectors"""
        self.index = faiss.IndexFlatL2(self.dim)