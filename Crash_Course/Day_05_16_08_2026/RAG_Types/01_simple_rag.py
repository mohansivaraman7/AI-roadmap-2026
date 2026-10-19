# Simple (Naive) RAG: embed docs -> FAISS retrieve top-k -> feed to LLM as context

from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np

documents = [
    "Machine Learning is a subset of Artificial Intelligence that learns patterns from data.",
    "Deep Learning uses multi-layer neural networks to learn complex representations.",
    "Python is a high-level programming language widely used for AI and data science.",
    "FAISS is a library for efficient similarity search over dense vectors.",
    "RAG stands for Retrieval-Augmented Generation, combining retrieval with an LLM.",
    "Transformers use self-attention to process sequences in parallel.",
    "A vector database stores embeddings and supports nearest-neighbor search.",
    "Fine-tuning adapts a pretrained model to a specific downstream task.",
]

embedder = SentenceTransformer("all-MiniLM-L6-v2")
generator = pipeline("text2text-generation", model="google/flan-t5-small")

doc_vectors = embedder.encode(documents)
index = faiss.IndexFlatL2(doc_vectors.shape[1])
index.add(np.array(doc_vectors, dtype=np.float32))


def retrieve(query, k=3):
    query_vector = embedder.encode([query])
    distances, indices = index.search(np.array(query_vector, dtype=np.float32), k)
    return [documents[i] for i in indices[0]]


def answer(query, k=3):
    context = retrieve(query, k)
    prompt = "Context:\n" + "\n".join(context) + f"\n\nQuestion: {query}\nAnswer:"
    response = generator(prompt, max_new_tokens=64)[0]["generated_text"]
    return response, context


if __name__ == "__main__":
    query = "What is RAG?"
    response, context = answer(query)

    print("Query:", query)
    print("\nRetrieved context:")
    for doc in context:
        print("-", doc)
    print("\nAnswer:", response)
