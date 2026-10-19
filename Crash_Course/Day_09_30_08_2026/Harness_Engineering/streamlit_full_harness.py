"""
streamlit_full_harness.py -- the end-user-facing app.
Imports everything from rag_engine.py (shared with mcp_server_full.py)
and wires it into a Streamlit chat UI + a LangChain agent (the TOOL LOOP
harness piece).

RUN WITH:
    streamlit run streamlit_full_harness.py

(Not `python streamlit_full_harness.py` -- see earlier notes on why.)
"""

from pathlib import Path
import uuid

import streamlit as st
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from rag_engine import (
    build_vector_db,
    build_reranker,
    make_tools,
    SYSTEM_PROMPT,
    groundedness_check,
    init_memory_db,
    save_turn,
    load_history,
)


# ============================================================
# CACHED SETUP -- built once per session, not on every message
# ============================================================

@st.cache_resource(show_spinner="Loading semantic FAISS index...")
def get_vector_db():
    return build_vector_db()


@st.cache_resource(show_spinner="Loading re-ranking model...")
def get_reranker():
    return build_reranker()


@st.cache_resource(show_spinner="Setting up the agent...")
def get_agent():
    vector_db, _ = get_vector_db()
    reranker = get_reranker()
    tools = make_tools(vector_db, reranker)          # 1. TOOL LOOPS
    llm = ChatOllama(model="llama3.1", temperature=0)
    return create_agent(llm, tools, system_prompt=SYSTEM_PROMPT)   # 2. FEEDFORWARD GUIDE goes in here


init_memory_db()   # 4. STATE & MEMORY -- ensure the SQLite table exists
agent_executor = get_agent()


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(page_title="Policy Assistant", page_icon="\U0001F4CB")
st.title("Policy Assistant \u2014 full harness")
st.caption("Semantic chunking + re-ranking + tools + groundedness check + persistent memory")


# ============================================================
# SESSION + PERSISTENT HISTORY
# ============================================================
# session_id identifies this browser session in the SQLite table. Unlike
# a plain st.session_state list, history saved here survives the whole
# app being closed and restarted -- real STATE & MEMORY, not just RAM.

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = load_history(st.session_state.session_id)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("groundedness_status") == "WARN":
            st.warning("Groundedness check flagged a possible unsupported number in this answer.", icon="\u26a0\ufe0f")


# ============================================================
# CHAT INPUT
# ============================================================

user_question = st.chat_input("Ask a question...")

if user_question:

    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question, "groundedness_status": None})
    save_turn(st.session_state.session_id, "user", user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = agent_executor.invoke(
                {"messages": [{"role": "user", "content": user_question}]}
            )
            answer = result["messages"][-1].content

            # 3. FEEDBACK SENSOR -- run AFTER the agent answers, before
            # showing it to the user. Pulls the tool outputs the agent
            # actually used this turn, to check the answer against them.
            tool_context = "\n".join(
                m.content for m in result["messages"]
                if getattr(m, "type", "") == "tool"
            )
            check = groundedness_check(answer, tool_context)

        st.markdown(answer)
        if check["status"] == "WARN":
            st.warning(f"Groundedness check: {check['reason']}", icon="\u26a0\ufe0f")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "groundedness_status": check["status"],
    })
    save_turn(st.session_state.session_id, "assistant", answer, check["status"])


# ============================================================
# SIDEBAR -- make the STATE & MEMORY layer visible, for teaching
# ============================================================

with st.sidebar:
    st.subheader("This session")
    st.caption(f"session_id: {st.session_state.session_id[:8]}...")
    st.caption(f"{len(st.session_state.messages)} messages saved to {Path('conversation_memory.db').absolute()}")
    st.caption("Reload this page (or close and reopen the app) -- this history will still be here, because it's read from SQLite, not just from memory.")
