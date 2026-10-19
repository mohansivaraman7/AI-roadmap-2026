# Agentic RAG: an LLM-driven loop decides ON ITS OWN whether retrieval is needed
# at all, and if the first retrieval isn't good enough, it rewrites the query and
# retrieves again (up to a max number of iterations) before answering.

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

MAX_ITERATIONS = 3
RELEVANCE_THRESHOLD = 0.35

embedder = SentenceTransformer("all-MiniLM-L6-v2")
generator = pipeline("text2text-generation", model="google/flan-t5-small")

doc_vectors = embedder.encode(documents, normalize_embeddings=True)
index = faiss.IndexFlatIP(doc_vectors.shape[1])
index.add(np.array(doc_vectors, dtype=np.float32))


def needs_retrieval(query):
    prompt = (
        f"Question: {query}\n"
        "Does answering this require looking up specific facts (yes/no)?"
    )
    reply = generator(prompt, max_new_tokens=5)[0]["generated_text"].strip().lower()
    return "yes" in reply or reply == ""  # default to retrieving when the model is unsure


def retrieve(query, k=3):
    query_vector = embedder.encode([query], normalize_embeddings=True)
    scores, indices = index.search(np.array(query_vector, dtype=np.float32), k)
    return [documents[i] for i in indices[0]], scores[0][0]


def rewrite_query(query, previous_attempts):
    prompt = (
        f"Original question: {query}\n"
        f"Previous search attempts that did not find a good answer: {previous_attempts}\n"
        "Rewrite the question with different, more specific keywords for a search engine."
    )
    return generator(prompt, max_new_tokens=32)[0]["generated_text"]


def answer(query):
    if not needs_retrieval(query):
        response = generator(query, max_new_tokens=64)[0]["generated_text"]
        return response, [], "answered_without_retrieval"

    context, best_score = [], -1.0
    attempts = []
    search_query = query

    for _ in range(MAX_ITERATIONS):
        context, best_score = retrieve(search_query)
        attempts.append(search_query)
        if best_score >= RELEVANCE_THRESHOLD:
            break
        search_query = rewrite_query(query, attempts)

    prompt = "Context:\n" + "\n".join(context) + f"\n\nQuestion: {query}\nAnswer:"
    response = generator(prompt, max_new_tokens=64)[0]["generated_text"]
    return response, context, f"retrieved_in_{len(attempts)}_attempt(s)"


if __name__ == "__main__":
    query = "What does RAG use to search for relevant documents?"
    response, context, strategy_used = answer(query)

    print("Query:", query)
    print("Strategy used:", strategy_used)
    print("\nFinal context:")
    for doc in context:
        print("-", doc)
    print("\nAnswer:", response)
