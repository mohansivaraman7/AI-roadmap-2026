"""
STREAMLIT VERSION of your agentic RAG script
==============================================
Same agent, same tools, same FAISS index -- only the interface changed.

What changed vs the PyCharm/terminal version:
  - input() / print()          -> st.chat_input() / st.chat_message()
  - while True loop             -> Streamlit reruns the whole script on every
                                    message, so chat history is kept in
                                    st.session_state instead of a Python list
                                    that lives only while the loop is running
  - Embeddings/FAISS/Agent load -> wrapped in @st.cache_resource so they load
    ONCE per session, not on every single message (Streamlit reruns the
    entire script top-to-bottom on every interaction, so without caching
    you'd reload the embedding model and rebuild the agent every message)

RUN THIS WITH:
    streamlit run streamlit_agentic_rag.py

(Do NOT run it with `python streamlit_agentic_rag.py` -- Streamlit apps
must be launched with the `streamlit run` command, not as a plain script.)

SETUP (same as before):
    pip install streamlit langchain langchain-ollama langchain-huggingface langchain-community faiss-cpu --break-system-packages
    ollama pull llama3.1
"""

from pathlib import Path

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama


# ============================================================
# CONFIGURATION  (unchanged)
# ============================================================

PDF_FILE = "company_policy.pdf"
DB_PATH = "faiss_index"


# ============================================================
# 1. BUILD / LOAD EVERYTHING ONCE  (cached across the whole session)
# ============================================================
# @st.cache_resource tells Streamlit: "run this function once, keep the
# result in memory, and hand back the same object on every rerun instead
# of recomputing it." This is the Streamlit equivalent of your
# if Path(DB_PATH).exists(): ... else: ... check, but it also prevents
# reloading the embedding model and rebuilding the agent every message.

@st.cache_resource(show_spinner="Loading embedding model and FAISS index...")
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if Path(DB_PATH).exists():
        vector_db = FAISS.load_local(
            DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        loader = PyPDFLoader(PDF_FILE)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        vector_db = FAISS.from_documents(chunks, embeddings)
        vector_db.save_local(DB_PATH)

    return vector_db


@st.cache_resource(show_spinner="Setting up the agent...")
def load_agent():
    vector_db = load_vector_db()
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # ----------------------------------------------------------
    # TOOLS  (identical to your PyCharm version)
    # ----------------------------------------------------------

    @tool
    def search_policy(query: str) -> str:
        """Search the company policy PDF and return the most relevant passages
        for the given question. Use this for ANY question about company rules,
        leave, WFH, expenses, IT, or compliance."""
        docs = retriever.invoke(query)
        if not docs:
            return "No matching policy content found."
        return "\n---\n".join(d.page_content for d in docs)

    @tool
    def raise_hr_ticket(employee_id: str, request_type: str, details: str) -> str:
        """Raise an HR ticket/request on behalf of the employee. Only call this
        AFTER the user has explicitly confirmed they want the action taken -
        never call this just because a policy question was asked."""
        ticket_id = "HR-20450"
        return f"Ticket {ticket_id} created for {employee_id}: [{request_type}] {details}"

    tools = [search_policy, raise_hr_ticket]

    # ----------------------------------------------------------
    # LLM  (must support tool calling)
    # ----------------------------------------------------------

    llm = ChatOllama(model="llama3.1", temperature=0)

    # ----------------------------------------------------------
    # AGENT
    # ----------------------------------------------------------
    # create_react_agent (from langgraph.prebuilt) is the current
    # replacement for the old create_tool_calling_agent + AgentExecutor
    # combo, which was removed from langchain.agents in recent
    # LangChain releases. It builds the same "LLM decides which tool
    # to call, runs it, feeds the result back" loop, just implemented
    # on top of LangGraph instead of the old AgentExecutor class.

    SYSTEM_PROMPT = """You are the company Policy Assistant.
Rules you must always follow:
- Answer policy questions using ONLY the search_policy tool's results. Never invent policy details.
- If the answer is not present in the retrieved content, say you don't know and suggest contacting HR.
- Never call raise_hr_ticket unless the user has clearly confirmed they want that action taken.
- Be concise and stick to what the retrieved content actually says.
"""

    return create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)


# ============================================================
# 2. PAGE SETUP
# ============================================================

st.set_page_config(page_title="Policy Assistant", page_icon="\U0001F4CB")
st.title("Agentic RAG Chatbot")
st.caption("Ask about company policy, or ask it to raise an HR request.")

agent_executor = load_agent()


# ============================================================
# 3. CHAT HISTORY
# ============================================================
# st.session_state persists across reruns WITHIN one browser session --
# this is what replaces the Python list you'd normally build up inside
# a while True loop.

if "messages" not in st.session_state:
    st.session_state.messages = []

# Redraw all previous messages on every rerun (Streamlit reruns the
# whole script top to bottom every time the user sends a new message)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# 4. CHAT INPUT  (replaces: question = input("You: "))
# ============================================================

user_question = st.chat_input("Ask a question...")

if user_question:

    # Show the user's message immediately
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    # Run the agent (replaces: result = agent_executor.invoke(...))
    # create_react_agent takes/returns a list of messages, not input/output
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = agent_executor.invoke(
                {"messages": [{"role": "user", "content": user_question}]}
            )
            answer = result["messages"][-1].content
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
