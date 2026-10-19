# Multi-Query RAG: an LLM rewrites the question into several variations,
# each variation retrieves docs, results are merged (deduped) before answering.

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


def generate_query_variations(query, n=3):
    prompt = f"Generate {n} different ways to ask this question, one per line:\n{query}"
    output = generator(prompt, max_new_tokens=64)[0]["generated_text"]
    variations = [line.strip("- ").strip() for line in output.split("\n") if line.strip()]
    variations.append(query)
    return list(dict.fromkeys(variations))  # dedupe, keep order


def retrieve(query, k=2):
    query_vector = embedder.encode([query])
    distances, indices = index.search(np.array(query_vector, dtype=np.float32), k)
    return [documents[i] for i in indices[0]]


def answer(query, k=2):
    variations = generate_query_variations(query)

    merged_context = []
    for q in variations:
        for doc in retrieve(q, k):
            if doc not in merged_context:
                merged_context.append(doc)

    prompt = "Context:\n" + "\n".join(merged_context) + f"\n\nQuestion: {query}\nAnswer:"
    response = generator(prompt, max_new_tokens=64)[0]["generated_text"]
    return response, variations, merged_context


if __name__ == "__main__":
    query = "How does retrieval-augmented generation work?"
    response, variations, context = answer(query)

    print("Query:", query)
    print("\nQuery variations used:")
    for v in variations:
        print("-", v)
    print("\nMerged context:")
    for doc in context:
        print("-", doc)
    print("\nAnswer:", response)
