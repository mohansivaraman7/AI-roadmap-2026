"""
policy_graph_app.py -- LangChain + LangGraph + LangSmith in ONE program
=========================================================================
Who does what here:

  LANGCHAIN  -> the building blocks: ChatOllama (the model), the @tool
                functions from rag_engine.py (FAISS + re-ranker + HR ticket
                + holiday API), message objects, ToolNode.
  LANGGRAPH  -> the control flow: an explicit graph (nodes + edges) with
                  * a human APPROVAL step before raise_hr_ticket runs
                  * a groundedness SENSOR that can send the agent back to
                    retry once if its answer has unsupported numbers
                  * a SQLite CHECKPOINTER so a conversation (and a paused
                    approval) survives restarts
  LANGSMITH  -> the visibility: every model call, tool call and graph node
                shows up as a trace tree at smith.langchain.com, with the
                groundedness check logged as its own step.

GRAPH:

    START -> agent --(wants sensitive tool)--> approval --yes--> tools -> agent
               |                                  |
               |                                  +--no--> agent (told "denied")
               +--(wants other tool)--> tools -> agent
               +--(final answer)--> groundedness --PASS--> END
                                         |
                                         +--WARN (first time)--> retry -> agent

SETUP (same as the rest of the course, plus LangGraph/LangSmith):
    pip install langgraph langgraph-checkpoint-sqlite langsmith langchain langchain-ollama
    ollama pull llama3.1
    python create_holiday_db.py
    uvicorn holiday_api:app --port 8000        (own terminal)

TURN ON LANGSMITH (optional -- the app works without it):
    1. Free account + API key: https://smith.langchain.com
    2. export LANGSMITH_API_KEY="lsv2_..."
    3. (optional) export LANGSMITH_PROJECT="policy-assistant"
    Tracing switches on automatically when the key is present.

RUN:
    python policy_graph_app.py                      # new conversation
    python policy_graph_app.py --thread my-chat     # resume a saved one
"""

import argparse
import os
import sqlite3
import uuid
from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from langsmith import traceable

from rag_engine import SYSTEM_PROMPT, groundedness_check

SENSITIVE_TOOLS = {"raise_hr_ticket"}   # tools that need a human "yes" first
MAX_RETRIES = 1                         # groundedness retries per user turn
CHECKPOINT_DB = "graph_checkpoints.db"


# ============================================================
# LANGSMITH SETUP
# ============================================================

def configure_langsmith() -> bool:
    """Turn tracing on only if an API key exists. Returns True if enabled."""
    if os.getenv("LANGSMITH_API_KEY"):
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", "policy-assistant")
        return True
    return False


# ============================================================
# LANGGRAPH STATE
# ============================================================
# State = the shared notebook every node reads from and writes to.
# `add_messages` means "append to the list" instead of overwriting it.

class PolicyState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    groundedness: dict
    retries: int


# ============================================================
# THE SENSOR, as its own traced step
# ============================================================

@traceable(name="groundedness_check", run_type="tool")
def traced_groundedness(answer: str, context: str) -> dict:
    """Same heuristic as rag_engine.groundedness_check, but wrapped so it
    appears as a named step in the LangSmith trace."""
    return groundedness_check(answer, context)


def _is_real_user_message(m) -> bool:
    return isinstance(m, HumanMessage) and not m.additional_kwargs.get("source")


def _current_turn_tool_output(messages) -> str:
    """Tool outputs from the CURRENT user turn only -- an old turn's tool
    results must not make a new answer look grounded."""
    last_user = max((i for i, m in enumerate(messages) if _is_real_user_message(m)), default=-1)
    return "\n".join(
        m.content for m in messages[last_user + 1:] if isinstance(m, ToolMessage)
    )


# ============================================================
# BUILD THE GRAPH
# ============================================================

def build_graph(llm, tools, checkpointer):
    llm_with_tools = llm.bind_tools(tools)

    # ---- node: agent (LangChain model decides: answer or call a tool)
    def agent_node(state: PolicyState):
        reply = llm_with_tools.invoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])
        return {"messages": [reply]}

    # ---- node: approval (LangGraph human-in-the-loop)
    def approval_node(state: PolicyState):
        last: AIMessage = state["messages"][-1]
        sensitive = [c for c in last.tool_calls if c["name"] in SENSITIVE_TOOLS]

        # interrupt() PAUSES the graph and saves state to the checkpointer.
        # The value we pass is what the caller shows to the human.
        decision = interrupt({
            "question": "Approve this action?",
            "actions": [{"tool": c["name"], "args": c["args"]} for c in sensitive],
        })

        if str(decision).strip().lower() in ("y", "yes", "approve", "approved"):
            return Command(goto="tools")

        # Denied: every tool_call needs a matching ToolMessage, otherwise the
        # model API rejects the next call. Answer ALL of them, not just the sensitive one.
        denials = [
            ToolMessage(
                content="The user denied this action. Do not retry it; tell them it was not done.",
                tool_call_id=c["id"],
                name=c["name"],
            )
            for c in last.tool_calls
        ]
        return Command(goto="agent", update={"messages": denials})

    # ---- node: tools (LangChain ToolNode runs the tool calls)
    tool_node = ToolNode(tools)

    # ---- node: groundedness (feedback sensor)
    def groundedness_node(state: PolicyState):
        answer = state["messages"][-1].content
        context = _current_turn_tool_output(state["messages"])
        return {"groundedness": traced_groundedness(answer, context)}

    # ---- node: retry (feed the sensor's complaint back to the agent)
    def retry_node(state: PolicyState):
        reason = state["groundedness"]["reason"]
        nudge = HumanMessage(
            content=(
                f"Your last answer failed a fact check: {reason} "
                "Search again if needed and answer using ONLY numbers that appear in the retrieved text."
            ),
            additional_kwargs={"source": "groundedness_sensor"},
        )
        return {"messages": [nudge], "retries": state["retries"] + 1}

    # ---- edges
    def route_after_agent(state: PolicyState):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            if any(c["name"] in SENSITIVE_TOOLS for c in last.tool_calls):
                return "approval"
            return "tools"
        return "groundedness"

    def route_after_groundedness(state: PolicyState):
        if state["groundedness"]["status"] == "WARN" and state["retries"] < MAX_RETRIES:
            return "retry"
        return END

    g = StateGraph(PolicyState)
    g.add_node("agent", agent_node)
    g.add_node("approval", approval_node)      # routes itself via Command
    g.add_node("tools", tool_node)
    g.add_node("groundedness", groundedness_node)
    g.add_node("retry", retry_node)

    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route_after_agent, ["approval", "tools", "groundedness"])
    g.add_edge("tools", "agent")
    g.add_conditional_edges("groundedness", route_after_groundedness, ["retry", END])
    g.add_edge("retry", "agent")
    # note: "approval" has no static outgoing edge -- its Command(goto=...) decides.

    return g.compile(checkpointer=checkpointer)


# ============================================================
# HELPERS FOR RUNNING THE GRAPH
# ============================================================

def make_checkpointer():
    """SQLite if available (survives restarts), otherwise in-memory."""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        return SqliteSaver(sqlite3.connect(CHECKPOINT_DB, check_same_thread=False))
    except ImportError:
        print("(langgraph-checkpoint-sqlite not installed -- using in-memory checkpointer)")
        return MemorySaver()


def run_config(thread_id: str) -> dict:
    return {
        "configurable": {"thread_id": thread_id},   # LangGraph: which conversation
        "run_name": "policy_turn",                  # LangSmith: name of the trace
        "tags": ["policy-assistant"],
        "metadata": {"thread_id": thread_id},
    }


def pending_interrupt(graph, config):
    """If the graph is paused waiting for a human, return what it asked."""
    for task in graph.get_state(config).tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None


def ask(graph, config, user_text: str, approve=input) -> dict:
    """One user turn. Handles the pause/resume loop for approvals.
    `approve` is a function that takes a prompt string and returns the
    human's answer (input() by default; tests pass a fake)."""
    graph.invoke(
        {"messages": [HumanMessage(user_text)], "retries": 0},   # retries reset every turn
        config,
    )
    while (request := pending_interrupt(graph, config)) is not None:
        print(f"\n[APPROVAL NEEDED] {request['question']}")
        for a in request["actions"]:
            print(f"   {a['tool']}({a['args']})")
        decision = approve("Type yes to approve, anything else to deny: ")
        graph.invoke(Command(resume=decision), config)
    return graph.get_state(config).values


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    from langchain_ollama import ChatOllama
    from rag_engine import build_reranker, build_vector_db, make_tools

    parser = argparse.ArgumentParser()
    parser.add_argument("--thread", default=None, help="conversation id; reuse to resume")
    args = parser.parse_args()
    thread_id = args.thread or f"chat-{uuid.uuid4().hex[:8]}"

    tracing = configure_langsmith()
    print(f"LangSmith tracing: {'ON -> project ' + os.environ['LANGSMITH_PROJECT'] if tracing else 'OFF (set LANGSMITH_API_KEY to enable)'}")
    print(f"Conversation thread: {thread_id}   (resume with --thread {thread_id})")

    vector_db, _ = build_vector_db()
    tools = make_tools(vector_db, build_reranker())
    llm = ChatOllama(model="llama3.1", temperature=0)

    graph = build_graph(llm, tools, make_checkpointer())
    config = run_config(thread_id)

    print("\nPolicy Assistant ready. Type 'quit' to exit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", ""):
            break
        state = ask(graph, config, question)
        print(f"\nAssistant: {state['messages'][-1].content}")
        check = state.get("groundedness") or {}
        if check.get("status") == "WARN":
            print(f"  [!] Groundedness warning: {check['reason']}")
        print()
