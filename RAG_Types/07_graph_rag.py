# Graph RAG: instead of (or in addition to) a vector index, facts are stored as a
# knowledge graph of (entity -> relation -> entity) triples. Retrieval walks the
# graph outward from entities mentioned in the query, so it can pull in facts that
# are connected by relationships even if they don't share vocabulary with the query.

from transformers import pipeline

knowledge_graph = {
    "Machine Learning": [("is_subset_of", "Artificial Intelligence"), ("uses", "Data")],
    "Deep Learning": [("is_subset_of", "Machine Learning"), ("uses", "Neural Networks")],
    "Neural Networks": [("inspired_by", "Human Brain"), ("used_in", "Deep Learning")],
    "Transformers": [("is_a", "Neural Network Architecture"), ("uses", "Self-Attention")],
    "RAG": [("combines", "Retrieval"), ("combines", "Generation"), ("uses", "Vector Database")],
    "Vector Database": [("stores", "Embeddings"), ("used_by", "RAG")],
    "FAISS": [("is_a", "Vector Database"), ("developed_by", "Meta")],
    "Fine-tuning": [("adapts", "Pretrained Model"), ("used_for", "Downstream Task")],
}

generator = pipeline("text2text-generation", model="google/flan-t5-small")


def find_mentioned_entities(query):
    return [entity for entity in knowledge_graph if entity.lower() in query.lower()]


def graph_walk(entity, depth=2):
    """Breadth-first walk collecting triples up to `depth` hops from `entity`."""
    visited = set()
    frontier = [entity]
    triples = []

    for _ in range(depth):
        next_frontier = []
        for node in frontier:
            if node in visited or node not in knowledge_graph:
                continue
            visited.add(node)
            for relation, target in knowledge_graph[node]:
                triples.append(f"{node} {relation.replace('_', ' ')} {target}")
                next_frontier.append(target)
        frontier = next_frontier

    return triples


def retrieve(query, depth=2):
    entities = find_mentioned_entities(query)
    context = []
    for entity in entities:
        context.extend(graph_walk(entity, depth))
    return list(dict.fromkeys(context))  # dedupe, keep order


def answer(query, depth=2):
    context = retrieve(query, depth)
    prompt = "Facts:\n" + "\n".join(context) + f"\n\nQuestion: {query}\nAnswer:"
    response = generator(prompt, max_new_tokens=64)[0]["generated_text"]
    return response, context


if __name__ == "__main__":
    query = "How does RAG relate to Deep Learning and Vector Database?"
    response, context = answer(query)

    print("Query:", query)
    print("\nGraph-derived facts:")
    for fact in context:
        print("-", fact)
    print("\nAnswer:", response)
