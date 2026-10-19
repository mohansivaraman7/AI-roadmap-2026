# Re-ranking RAG: FAISS retrieves a wide candidate set (recall-focused),
# then a cross-encoder re-scores each (query, doc) pair for precision before generation.

from sentence_transformers import SentenceTransformer, CrossEncoder
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
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
generator = pipeline("text2text-generation", model="google/flan-t5-small")

doc_vectors = embedder.encode(documents)
index = faiss.IndexFlatL2(doc_vectors.shape[1])
index.add(np.array(doc_vectors, dtype=np.float32))


def retrieve_candidates(query, candidate_k=6):
    query_vector = embedder.encode([query])
    _, indices = index.search(np.array(query_vector, dtype=np.float32), candidate_k)
    return [documents[i] for i in indices[0]]


def rerank(query, candidates, top_k=3):
    pairs = [[query, doc] for doc in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in ranked[:top_k]]


def answer(query, candidate_k=6, top_k=3):
    candidates = retrieve_candidates(query, candidate_k)
    context = rerank(query, candidates, top_k)
    prompt = "Context:\n" + "\n".join(context) + f"\n\nQuestion: {query}\nAnswer:"
    response = generator(prompt, max_new_tokens=64)[0]["generated_text"]
    return response, candidates, context


if __name__ == "__main__":
    query = "How is a vector database used for search?"
    response, candidates, context = answer(query)

    print("Query:", query)
    print("\nInitial candidates (vector search):")
    for doc in candidates:
        print("-", doc)
    print("\nRe-ranked top context:")
    for doc in context:
        print("-", doc)
    print("\nAnswer:", response)
