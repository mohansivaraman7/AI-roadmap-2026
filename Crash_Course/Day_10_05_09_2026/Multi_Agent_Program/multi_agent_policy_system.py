"""
multi_agent_policy_system.py -- GENUINE multi-agent AI, for direct
comparison against streamlit_full_harness.py / rag_chat_llm.py, which
are both SINGLE-agent, multi-TOOL systems.

THE KEY DIFFERENCE, made concrete:

  streamlit_full_harness.py (single agent, multiple tools)
  ----------------------------------------------------------
      ONE create_agent() call
      ONE llm.invoke() per turn (plus tool-call rounds)
      Tools are plain Python functions -- no reasoning inside them
      search_policy() just does retriever.invoke(), no LLM call at all

  multi_agent_policy_system.py (THIS file -- true multi-agent)
  ----------------------------------------------------------
      FOUR separate create_agent() calls: 3 specialists + 1 supervisor
      Each specialist has its OWN llm.invoke(), its OWN system prompt,
      and reasons independently before answering
      The supervisor never answers policy/HR/holiday questions itself --
      it only DECIDES which specialist to hand the question to

Architecture:

                     ┌─────────────────┐
        question --> │   SUPERVISOR    │
                     │  (routes only,  │
                     │  never answers  │
                     │  directly)      │
                     └───┬────┬────┬───┘
                         │    │    │
              ┌──────────┘    │    └──────────┐
              v                v                v
      ┌───────────────┐ ┌──────────────┐ ┌────────────────┐
      │ POLICY AGENT   │ │  HR AGENT    │ │ HOLIDAY AGENT  │
      │ (own LLM call) │ │(own LLM call)│ │ (own LLM call) │
      │ tool:          │ │ tool:        │ │ tool:          │
      │ search_policy  │ │raise_hr_     │ │check_public_   │
      │                │ │ticket        │ │holiday         │
      └───────────────┘ └──────────────┘ └────────────────┘

Each specialist is a REAL agent (its own create_agent call), not just a
plain function -- that's what makes this genuinely multi-agent instead
of single-agent-multi-tool.

SETUP: same as streamlit_full_harness.py
    pip install -r requirements.txt langchain-experimental sentence-transformers \
                fastapi uvicorn --break-system-packages
    ollama pull llama3.1
    python create_holiday_db.py
    uvicorn holiday_api:app --port 8000   (leave running)

RUN:
    python multi_agent_policy_system.py
"""

from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from rag_engine import build_vector_db, build_reranker, make_tools


# ============================================================
# 1. SHARED SETUP -- reused from rag_engine.py, not duplicated
# ============================================================

print("Loading semantic FAISS index...")
vector_db, _ = build_vector_db()

print("Loading re-ranking model...")
reranker = build_reranker()

print("Building base tools...")
search_policy, raise_hr_ticket, check_public_holiday = make_tools(vector_db, reranker)

llm = ChatOllama(model="llama3.1", temperature=0)


# ============================================================
# 2. THREE INDEPENDENT SPECIALIST AGENTS
# ============================================================
# Each of these is a FULL agent -- its own create_agent() call, its own
# system prompt, its own reasoning loop. This is the structural
# difference from the single-agent version: search_policy here is
# handed to a dedicated agent, not called as a bare function directly
# by one shared agent.

policy_agent = create_agent(
    llm,
    [search_policy],
    prompt=(
        "You are a company policy research specialist. Answer ONLY using "
        "the search_policy tool's results. Never invent policy details. "
        "If the answer isn't in the retrieved content, say you don't know."
    ),
)

hr_agent = create_agent(
    llm,
    [raise_hr_ticket],
    prompt=(
        "You are an HR ticketing specialist. Only call raise_hr_ticket "
        "when the user has clearly confirmed they want a request submitted. "
        "Otherwise, ask for the missing details (employee ID, request type)."
    ),
)

holiday_agent = create_agent(
    llm,
    [check_public_holiday],
    prompt=(
        "You are a public-holiday calendar specialist. Use check_public_holiday "
        "to answer questions about whether a specific date is a holiday, in a "
        "specific country. Ask for a date and country code if either is missing."
    ),
)


# ============================================================
# 3. WRAP EACH SPECIALIST AS A TOOL FOR THE SUPERVISOR
# ============================================================
# This is the mechanism that makes hand-off possible: from the
# supervisor's point of view, calling a specialist looks exactly like
# calling any other tool -- but under the hood, it triggers a SEPARATE
# agent's SEPARATE reasoning loop, with its own LLM call.

@tool
def ask_policy_specialist(question: str) -> str:
    """Ask the company policy specialist about leave, WFH, expenses,
    IT, or compliance questions."""
    print(f"    [supervisor -> policy_agent]  {question}")
    result = policy_agent.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content


@tool
def ask_hr_specialist(question: str) -> str:
    """Ask the HR ticketing specialist to raise or discuss an HR request/ticket."""
    print(f"    [supervisor -> hr_agent]      {question}")
    result = hr_agent.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content


@tool
def ask_holiday_specialist(question: str) -> str:
    """Ask the holiday calendar specialist whether a specific date is a
    public holiday in a specific country."""
    print(f"    [supervisor -> holiday_agent] {question}")
    result = holiday_agent.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content


# ============================================================
# 4. THE SUPERVISOR -- routes only, never answers directly
# ============================================================
# Notice: the supervisor has NO access to search_policy, raise_hr_ticket,
# or check_public_holiday directly. Its only tools are the three
# specialists above. It cannot do the work itself -- it can only decide
# WHO should.

SUPERVISOR_PROMPT = """You are a routing supervisor for a company assistant.
You do NOT answer questions yourself. For every question, decide which
ONE specialist should handle it, call that specialist, and relay their
answer back to the user:
  - ask_policy_specialist  for leave, WFH, expenses, IT, compliance questions
  - ask_hr_specialist      for raising or discussing HR tickets/requests
  - ask_holiday_specialist for public holiday date questions
If a question needs more than one specialist, call each in turn and
combine their answers. Never fabricate an answer without consulting a
specialist first.
"""

supervisor = create_agent(
    llm,
    [ask_policy_specialist, ask_hr_specialist, ask_holiday_specialist],
    prompt=SUPERVISOR_PROMPT,
)


# ============================================================
# 5. CHAT LOOP
# ============================================================

print("\n======================================")
print("   MULTI-AGENT POLICY SYSTEM")
print("======================================")
print("Ask a question. Watch the [supervisor -> ...] lines below --")
print("each one is a SEPARATE agent, with its own LLM reasoning call.")
print("Type 'exit' to stop.\n")

while True:
    question = input("You: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    result = supervisor.invoke({"messages": [{"role": "user", "content": question}]})
    answer = result["messages"][-1].content

    print("\nAI:", answer)
    print("\n" + "=" * 60)
