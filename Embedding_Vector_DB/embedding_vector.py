from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Documents
documents = [
    "Machine Learning is a subset of AI",
    "Python is a programming language",
    "Deep Learning uses neural networks"
]

# Convert documents to vectors
vectors = model.encode(documents)

# Create FAISS index
dimension = vectors.shape[1]

index = faiss.IndexFlatL2(dimension)

# Add vectors to FAISS
index.add(np.array(vectors, dtype=np.float32))

print("Total documents stored:", index.ntotal)