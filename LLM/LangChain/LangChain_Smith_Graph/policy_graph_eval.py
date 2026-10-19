"""
policy_graph_eval.py -- LangSmith EVALUATION of the policy graph
=================================================================
Tracing (policy_graph_app.py) shows you what happened on ONE run.
Evaluation answers "is the whole system still good?" across MANY questions,
and lets you compare experiments after you change a prompt, model or chunker.

What this does:
  1. Creates a small dataset "policy-assistant-qa" in LangSmith (once).
  2. Runs every question through the real graph.
  3. Scores each run with 3 evaluators:
       - answer_contains_fact : does the answer include the expected fact?
       - used_expected_tool   : did the agent call the right tool?
       - grounded             : did the groundedness sensor pass?
  4. Prints a link to the experiment so you can compare runs side by side.

!! EDIT the EXAMPLES below so the expected facts match YOUR company_policy.pdf.

RUN (needs LANGSMITH_API_KEY, Ollama running, holiday_api running):
    export LANGSMITH_API_KEY="lsv2_..."
    python policy_graph_eval.py
"""

import os
import sys
import uuid

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langsmith import Client, evaluate

DATASET_NAME = "policy-assistant-qa"

# Edit these to match your PDF. expected_fact is a lowercase substring.
EXAMPLES = [
    {"question": "How many days of annual leave do employees get?",
     "expected_fact": "", "expected_tool": "search_policy"},
    {"question": "Is 2026-01-26 a public holiday in India?",
     "expected_fact": "republic day", "expected_tool": "check_public_holiday"},
    {"question": "What is the policy on working from home?",
     "expected_fact": "", "expected_tool": "search_policy"},
]


# ---------- the thing being tested ----------

def make_target(graph):
    def target(inputs: dict) -> dict:
        config = {"configurable": {"thread_id": f"eval-{uuid.uuid4().hex[:8]}"}}
        out = graph.invoke(
            {"messages": [HumanMessage(inputs["question"])], "retries": 0}, config
        )
        msgs = out["messages"]
        tools_used = [
            c["name"] for m in msgs if isinstance(m, AIMessage) for c in (m.tool_calls or [])
        ]
        return {
            "answer": msgs[-1].content,
            "tools_used": tools_used,
            "groundedness": (out.get("groundedness") or {}).get("status", "PASS"),
        }
    return target


# ---------- evaluators: (run outputs, reference outputs) -> score ----------

def answer_contains_fact(outputs: dict, reference_outputs: dict) -> dict:
    fact = (reference_outputs.get("expected_fact") or "").lower()
    if not fact:   # nothing to check for this example
        return {"key": "answer_contains_fact", "score": 1}
    return {"key": "answer_contains_fact", "score": int(fact in outputs["answer"].lower())}


def used_expected_tool(outputs: dict, reference_outputs: dict) -> dict:
    return {
        "key": "used_expected_tool",
        "score": int(reference_outputs["expected_tool"] in outputs["tools_used"]),
    }


def grounded(outputs: dict) -> dict:
    return {"key": "grounded", "score": int(outputs["groundedness"] == "PASS")}


# ---------- main ----------

def ensure_dataset(client: Client):
    if client.has_dataset(dataset_name=DATASET_NAME):
        return
    ds = client.create_dataset(DATASET_NAME, description="Policy assistant regression questions")
    client.create_examples(
        dataset_id=ds.id,
        inputs=[{"question": e["question"]} for e in EXAMPLES],
        outputs=[{"expected_fact": e["expected_fact"], "expected_tool": e["expected_tool"]} for e in EXAMPLES],
    )
    print(f"Created dataset '{DATASET_NAME}' with {len(EXAMPLES)} examples.")


if __name__ == "__main__":
    if not os.getenv("LANGSMITH_API_KEY"):
        sys.exit("Set LANGSMITH_API_KEY first (free key at https://smith.langchain.com).")

    from langchain_ollama import ChatOllama
    from langgraph.checkpoint.memory import MemorySaver
    from policy_graph_app import build_graph
    from rag_engine import build_reranker, build_vector_db, make_tools

    vector_db, _ = build_vector_db()
    tools = make_tools(vector_db, build_reranker())
    llm = ChatOllama(model="llama3.1", temperature=0)
    graph = build_graph(llm, tools, MemorySaver())

    client = Client()
    ensure_dataset(client)

    results = evaluate(
        make_target(graph),
        data=DATASET_NAME,
        evaluators=[answer_contains_fact, used_expected_tool, grounded],
        experiment_prefix="policy-graph",
        metadata={"model": "llama3.1", "chunking": "semantic", "reranker": "ms-marco-MiniLM"},
        max_concurrency=1,   # local Ollama handles one request at a time
    )
    print("\nDone. Open the experiment link above in LangSmith to compare runs.")
