# HyDE (Hypothetical Document Embeddings): the LLM first writes a hypothetical
# answer to the question, and THAT answer's embedding (not the raw question) is
# used to search the vector store, since it reads more like the target documents.

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


def generate_hypothetical_document(query):
    prompt = f"Write one short sentence that would answer this question: {query}"
    return generator(prompt, max_new_tokens=48)[0]["generated_text"]


def retrieve(text, k=3):
    vector = embedder.encode([text])
    _, indices = index.search(np.array(vector, dtype=np.float32), k)
    return [documents[i] for i in indices[0]]


def answer(query, k=3):
    hypothetical_doc = generate_hypothetical_document(query)
    context = retrieve(hypothetical_doc, k)
    prompt = "Context:\n" + "\n".join(context) + f"\n\nQuestion: {query}\nAnswer:"
    response = generator(prompt, max_new_tokens=64)[0]["generated_text"]
    return response, hypothetical_doc, context


if __name__ == "__main__":
    query = "What technique adapts a pretrained model for a new task?"
    response, hypothetical_doc, context = answer(query)

    print("Query:", query)
    print("\nHypothetical document (used for retrieval):")
    print(hypothetical_doc)
    print("\nRetrieved context:")
    for doc in context:
        print("-", doc)
    print("\nAnswer:", response)
