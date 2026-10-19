"""
rag_engine.py -- SHARED CORE, used by BOTH streamlit_full_harness.py and
mcp_server_full.py. This is deliberate: instead of duplicating the FAISS
setup, tools, and prompt in two files (like your earlier scripts did),
both front-ends import from this ONE place. Change a tool here, both
apps get the fix.

This file implements every piece from the "Core Components of a Harness"
breakdown, each clearly labeled below:

    1. TOOL LOOPS        -> search_policy, raise_hr_ticket, check_public_holiday
    2. FEEDFORWARD GUIDE  -> SYSTEM_PROMPT
    3. FEEDBACK SENSOR    -> groundedness_check()
    4. STATE & MEMORY     -> SQLite-backed conversation log (persists across
                             restarts, unlike a plain Python list or
                             st.session_state alone)

SETUP:
    pip install -r requirements.txt --break-system-packages
    pip install langchain-experimental sentence-transformers fastapi uvicorn --break-system-packages
    ollama pull llama3.1

BEFORE FIRST RUN:
    python create_holiday_db.py
    uvicorn holiday_api:app --port 8000   (leave running in its own terminal)
"""

import sqlite3
import re
from pathlib import Path
from datetime import datetime, timezone

import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from sentence_transformers import CrossEncoder


# ============================================================
# CONFIGURATION
# ============================================================

PDF_FILE = "company_policy.pdf"
DB_PATH = "faiss_index_semantic"          # semantic chunks -> own index, never mix with character-based ones
MEMORY_DB = "conversation_memory.db"       # STATE & MEMORY lives here, not just in RAM
HOLIDAY_API_URL = "http://127.0.0.1:8000/holidays"

CANDIDATE_POOL_SIZE = 8    # FAISS over-fetches this many before re-ranking
FINAL_TOP_K = 3            # re-ranker keeps only this many


# ============================================================
# INGESTION -- semantic chunking (built once, cached by caller)
# ============================================================

def build_vector_db():
    """Loads or builds the FAISS index using SemanticChunker instead of
    a fixed character count -- chunks break where the TOPIC shifts, not
    where a character counter runs out."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if Path(DB_PATH).exists():
        vector_db = FAISS.load_local(
            DB_PATH, embeddings, allow_dangerous_deserialization=True
        )
    else:
        loader = PyPDFLoader(PDF_FILE)
        documents = loader.load()

        splitter = SemanticChunker(embeddings)
        chunks = splitter.split_documents(documents)

        vector_db = FAISS.from_documents(chunks, embeddings)
        vector_db.save_local(DB_PATH)

    return vector_db, embeddings


def build_reranker():
    """Cross-encoder used to re-score FAISS's candidate pool. Loading this
    model is slow-ish, so the caller (Streamlit/MCP) should cache it."""
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ============================================================
# 1. TOOL LOOPS -- the functions an agent can actually call
# ============================================================

def make_tools(vector_db, reranker):
    """Builds the 3 tools, closing over this process's vector_db/reranker.
    Returns a plain list of @tool-decorated functions -- usable directly
    by a LangChain agent (Streamlit) OR wrapped again with @mcp.tool()
    (MCP server) without rewriting the retrieval/re-ranking logic twice.
    """
    retriever = vector_db.as_retriever(search_kwargs={"k": CANDIDATE_POOL_SIZE})

    @tool
    def search_policy(query: str) -> str:
        """Search the company policy document and return the most relevant
        passages for the given question. Use this for any question about
        company rules, leave, WFH, expenses, IT, or compliance."""
        candidates = retriever.invoke(query)
        if not candidates:
            return "No matching policy content found."

        pairs = [[query, doc.page_content] for doc in candidates]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in ranked[:FINAL_TOP_K]]

        return "\n---\n".join(d.page_content for d in top_docs)

    @tool
    def raise_hr_ticket(employee_id: str, request_type: str, details: str) -> str:
        """Raise an HR ticket/request on behalf of the employee. Only call
        this AFTER the user has explicitly confirmed they want the action
        taken -- never call this just because a policy question was asked."""
        ticket_id = "HR-20450"
        return f"Ticket {ticket_id} created for {employee_id}: [{request_type}] {details}"

    @tool
    def check_public_holiday(date: str, country_code: str = "IN") -> str:
        """Check whether a given date is a public holiday in a specific
        country. date must be YYYY-MM-DD. country_code is a 2-letter ISO
        code (e.g. IN, US, GB)."""
        try:
            response = requests.get(
                HOLIDAY_API_URL,
                params={"date": date, "country_code": country_code.upper()},
                timeout=5,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            return (f"Could not reach the local holiday API: {e}. "
                    f"Make sure it's running: uvicorn holiday_api:app --port 8000")

        result = response.json()
        if result["is_holiday"]:
            return f"Yes, {date} is a public holiday in {country_code.upper()}: {result['name']} ({result['local_name']})."
        return f"No, {date} is not listed as a public holiday in {country_code.upper()}."

    return [search_policy, raise_hr_ticket, check_public_holiday]


# ============================================================
# 2. FEEDFORWARD GUIDE -- steers the agent BEFORE it acts
# ============================================================
# This is the same role as a CLAUDE.md / AGENTS.md file: standing
# instructions the agent reads at the start of every turn.

SYSTEM_PROMPT = """You are the company Policy Assistant.
Rules you must always follow:
- Answer policy questions using ONLY the search_policy tool's results. Never invent policy details.
- If the answer is not present in the retrieved content, say you don't know and suggest contacting HR.
- Never call raise_hr_ticket unless the user has clearly confirmed they want that action taken.
- Be concise and stick to what the retrieved content actually says.
"""


# ============================================================
# 3. FEEDBACK SENSOR -- catches mistakes AFTER the agent acts
# ============================================================
# A real production system would use something like RAGAS (faithfulness,
# context precision/recall) and likely a second LLM call as a judge.
# This is a lightweight, dependency-free stand-in for teaching purposes:
# it checks whether the numbers the agent's answer relies on actually
# appear somewhere in the retrieved context -- a simple hallucinated-
# number detector, not a full groundedness evaluator.

def groundedness_check(answer: str, context: str) -> dict:
    """Returns a dict describing whether the answer's numeric claims are
    backed by the retrieved context. This is a heuristic, not a real
    evaluation framework -- it exists to make 'feedback sensors' concrete,
    not to replace RAGAS-style evaluation in a real deployment."""
    answer_numbers = set(re.findall(r"\b\d+\b", answer))
    context_numbers = set(re.findall(r"\b\d+\b", context))

    unsupported = answer_numbers - context_numbers

    if not answer_numbers:
        return {"status": "PASS", "reason": "No numeric claims to verify."}
    if unsupported:
        return {
            "status": "WARN",
            "reason": f"Answer contains number(s) {sorted(unsupported)} not found in the retrieved context -- possible hallucination.",
        }
    return {"status": "PASS", "reason": "All numeric claims in the answer appear in the retrieved context."}


# ============================================================
# 4. STATE & MEMORY -- persists across restarts, not just in RAM
# ============================================================
# st.session_state (used in earlier scripts) only lives as long as the
# browser tab/session does. This SQLite table survives the app being
# closed and reopened entirely -- a real (if simple) example of external
# memory, the same role a database plays for a long-running agent.

def init_memory_db():
    connection = sqlite3.connect(MEMORY_DB)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS conversation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            groundedness_status TEXT,
            timestamp TEXT NOT NULL
        )
    """)
    connection.commit()
    connection.close()


def save_turn(session_id: str, role: str, content: str, groundedness_status: str = None):
    connection = sqlite3.connect(MEMORY_DB)
    connection.execute(
        "INSERT INTO conversation (session_id, role, content, groundedness_status, timestamp) VALUES (?, ?, ?, ?, ?)",
        (session_id, role, content, groundedness_status, datetime.now(timezone.utc).isoformat()),
    )
    connection.commit()
    connection.close()


def load_history(session_id: str):
    connection = sqlite3.connect(MEMORY_DB)
    cursor = connection.execute(
        "SELECT role, content, groundedness_status FROM conversation WHERE session_id = ? ORDER BY id",
        (session_id,),
    )
    rows = cursor.fetchall()
    connection.close()
    return [{"role": r, "content": c, "groundedness_status": g} for r, c, g in rows]
