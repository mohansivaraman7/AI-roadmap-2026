# Hybrid Search RAG: combines keyword search (BM25) with vector search (FAISS),
# then merges scores so both exact keyword matches and semantic matches count.

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
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

# Vector index
doc_vectors = embedder.encode(documents)
index = faiss.IndexFlatL2(doc_vectors.shape[1])
index.add(np.array(doc_vectors, dtype=np.float32))

# Keyword index
tokenized_docs = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)


def normalize(scores):
    scores = np.array(scores, dtype=np.float32)
    if scores.max() == scores.min():
        return np.zeros_like(scores)
    return (scores - scores.min()) / (scores.max() - scores.min())


def hybrid_retrieve(query, k=3, alpha=0.5):
    query_vector = embedder.encode([query])
    distances, indices = index.search(np.array(query_vector, dtype=np.float32), len(documents))

    # smaller L2 distance -> higher similarity, remapped to original document order
    vector_scores = np.zeros(len(documents), dtype=np.float32)
    for rank, doc_idx in enumerate(indices[0]):
        vector_scores[doc_idx] = -distances[0][rank]

    keyword_scores = bm25.get_scores(query.lower().split())

    combined = alpha * normalize(vector_scores) + (1 - alpha) * normalize(keyword_scores)
    top_k = np.argsort(combined)[::-1][:k]
    return [documents[i] for i in top_k]


def answer(query, k=3):
    context = hybrid_retrieve(query, k)
    prompt = "Context:\n" + "\n".join(context) + f"\n\nQuestion: {query}\nAnswer:"
    response = generator(prompt, max_new_tokens=64)[0]["generated_text"]
    return response, context


if __name__ == "__main__":
    query = "What library does FAISS provide for vectors?"
    response, context = answer(query)

    print("Query:", query)
    print("\nHybrid retrieved context:")
    for doc in context:
        print("-", doc)
    print("\nAnswer:", response)
