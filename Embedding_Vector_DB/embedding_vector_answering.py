from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Step 1: Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 2: Documents that we want to store
documents = [
    "Machine Learning is a subset of AI",
    "Python is a programming language",
    "Deep Learning uses neural networks"
]

# Step 3: Convert documents into vectors
vectors = model.encode(documents)

print("Vector Shape:", vectors.shape)

# Step 4: Create FAISS index
# vectors.shape[1] = number of columns in embedding
dimension = vectors.shape[1]

index = faiss.IndexFlatL2(dimension)

# Step 5: Add document vectors into FAISS
index.add(np.array(vectors, dtype=np.float32))

print("Documents stored:", index.ntotal)

# -------------------------
# USER QUERY
# -------------------------

query = "What is Artificial Intelligence?"

# Step 6: Convert query into vector
query_vector = model.encode([query])

# Step 7: Search top 1 matching document
distances, indices = index.search(
    np.array(query_vector, dtype=np.float32),
    k=1
)

print("Distance:", distances)
print("Index Found:", indices)

# Step 8: Retrieve actual document
matched_document = documents[indices[0][0]]

print("Matched Document:")
print(matched_document)