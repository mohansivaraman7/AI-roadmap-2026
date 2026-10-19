# Corrective RAG (CRAG): retrieved docs are graded for relevance (via similarity
# score against the query). If the best match is weak, the retrieval is treated as
# unreliable and the system falls back to keyword search over the full corpus
# instead of trusting a low-confidence vector match.

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

RELEVANCE_THRESHOLD = 0.35  # cosine similarity below this is considered "not relevant"

embedder = SentenceTransformer("all-MiniLM-L6-v2")
generator = pipeline("text2text-generation", model="google/flan-t5-small")

doc_vectors = embedder.encode(documents, normalize_embeddings=True)
index = faiss.IndexFlatIP(doc_vectors.shape[1])  # inner product == cosine sim on normalized vectors
index.add(np.array(doc_vectors, dtype=np.float32))


def vector_retrieve(query, k=3):
    query_vector = embedder.encode([query], normalize_embeddings=True)
    scores, indices = index.search(np.array(query_vector, dtype=np.float32), k)
    return [documents[i] for i in indices[0]], scores[0]


def keyword_fallback(query, k=3):
    query_terms = set(query.lower().split())
    scored = []
    for doc in documents:
        overlap = len(query_terms & set(doc.lower().split()))
        scored.append((doc, overlap))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored[:k]]


def grade_and_retrieve(query, k=3):
    docs, scores = vector_retrieve(query, k)
    if scores[0] < RELEVANCE_THRESHOLD:
        return keyword_fallback(query, k), "keyword_fallback"
    return docs, "vector_search"


def answer(query, k=3):
    context, strategy_used = grade_and_retrieve(query, k)
    prompt = "Context:\n" + "\n".join(context) + f"\n\nQuestion: {query}\nAnswer:"
    response = generator(prompt, max_new_tokens=64)[0]["generated_text"]
    return response, context, strategy_used


if __name__ == "__main__":
    query = "Tell me about GPU cooling solutions"  # not in the corpus -> triggers fallback
    response, context, strategy_used = answer(query)

    print("Query:", query)
    print("Strategy used:", strategy_used)
    print("\nContext:")
    for doc in context:
        print("-", doc)
    print("\nAnswer:", response)
