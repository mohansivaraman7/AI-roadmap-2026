"""
STREAMLIT AGENTIC RAG -- WITH SEMANTIC CHUNKING + RE-RANKING
==============================================================
Same agent, same tools, same Streamlit chat UI as before.
TWO things upgraded vs your previous version:

  1. CHUNKING: RecursiveCharacterTextSplitter (character-count based)
     ->  SemanticChunker (splits where the MEANING actually shifts,
         not where a fixed character count runs out)

  2. RETRIEVAL: plain FAISS similarity search only
     ->  FAISS retrieves a wider pool of candidates first, then a
         CROSS-ENCODER re-ranks those candidates by how well each one
         actually answers the question, and only the top few (after
         re-ranking) are handed to the agent

Why both matter together: semantic chunking gives you cleaner,
topic-coherent chunks going IN. Re-ranking gives you more accurate
chunk selection coming OUT (at query time). Chunking quality and
retrieval quality are two separate problems -- this script upgrades
both.

RUN THIS WITH:
    streamlit run streamlit_agentic_rag_semantic_rerank.py

SETUP (adds two new packages vs your previous script):
    pip install streamlit langchain langchain-ollama langchain-huggingface \
                langchain-community langchain-experimental sentence-transformers \
                faiss-cpu --break-system-packages

    ollama pull llama3.1
"""

from pathlib import Path
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

# NEW: semantic chunking splitter (meaning-based, not character-count based)
from langchain_experimental.text_splitter import SemanticChunker

# NEW: cross-encoder for re-ranking retrieved chunks
from sentence_transformers import CrossEncoder


# ============================================================
# CONFIGURATION
# ============================================================

PDF_FILE = "company_policy.pdf"

# NOTE: new DB_PATH -- semantic chunking produces DIFFERENT chunks than
# RecursiveCharacterTextSplitter did, so this must NOT reuse your old
# faiss_index/ folder. Using a new folder forces a fresh build.
DB_PATH = "faiss_index_semantic"

CANDIDATE_POOL_SIZE = 8   # how many chunks FAISS retrieves BEFORE re-ranking
FINAL_TOP_K = 3           # how many chunks survive AFTER re-ranking


# ============================================================
# 1. BUILD / LOAD EVERYTHING ONCE  (cached across the whole session)
# ============================================================

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

        # ----------------------------------------------------------
        # SEMANTIC CHUNKING (replaces RecursiveCharacterTextSplitter)
        # ----------------------------------------------------------
        # SemanticChunker embeds sentences, measures how similar each
        # sentence is to the next, and starts a NEW chunk exactly where
        # the topic meaningfully shifts. No fixed chunk_size/chunk_overlap
        # at all -- a chunk is as long as the topic naturally runs.
        splitter = SemanticChunker(embeddings)
        chunks = splitter.split_documents(documents)

        vector_db = FAISS.from_documents(chunks, embeddings)
        vector_db.save_local(DB_PATH)

    return vector_db, embeddings


@st.cache_resource(show_spinner="Loading re-ranking model...")
def load_reranker():
    # A cross-encoder scores a (question, chunk) PAIR directly, which is
    # more accurate than plain embedding similarity -- but too slow to run
    # against every chunk in a big database. So the usual pattern is:
    #   1. FAISS (fast, approximate) narrows thousands of chunks down to a
    #      small candidate pool (CANDIDATE_POOL_SIZE)
    #   2. Cross-encoder (slow, accurate) re-scores just that small pool
    #      and keeps only the real top matches (FINAL_TOP_K)
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


@st.cache_resource(show_spinner="Setting up the agent...")
def load_agent():
    vector_db, embeddings = load_vector_db()
    reranker = load_reranker()

    # Retriever now pulls a WIDER pool than before (8 instead of 3) --
    # re-ranking needs candidates to choose from, so we deliberately
    # over-fetch here and narrow down after scoring.
    retriever = vector_db.as_retriever(search_kwargs={"k": CANDIDATE_POOL_SIZE})

    # ----------------------------------------------------------
    # TOOLS
    # ----------------------------------------------------------

    @tool
    def search_policy(query: str) -> str:
        """Search the company policy PDF and return the most relevant passages
        for the given question. Use this for ANY question about company rules,
        leave, WFH, expenses, IT, or compliance."""

        # Step 1: FAISS similarity search -> wide candidate pool
        candidates = retriever.invoke(query)
        if not candidates:
            return "No matching policy content found."

        # Step 2: RE-RANKING -- score each (query, chunk) pair with the
        # cross-encoder. Unlike FAISS's embedding similarity (which compares
        # two vectors that were embedded SEPARATELY), a cross-encoder reads
        # the question and the chunk TOGETHER in one pass, which usually
        # gives a more accurate relevance judgment.
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = reranker.predict(pairs)

        # Step 3: sort candidates by re-ranked score, keep only the best few
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in ranked[:FINAL_TOP_K]]

        return "\n---\n".join(d.page_content for d in top_docs)

    @tool
    def raise_hr_ticket(employee_id: str, request_type: str, details: str) -> str:
        """Raise an HR ticket/request on behalf of the employee. Only call this
        AFTER the user has explicitly confirmed they want the action taken -
        never call this just because a policy question was asked."""
        ticket_id = "HR-20450"
        return f"Ticket {ticket_id} created for {employee_id}: [{request_type}] {details}"

    tools = [search_policy, raise_hr_ticket]

    # ----------------------------------------------------------
    # LLM + AGENT
    # ----------------------------------------------------------

    llm = ChatOllama(model="llama3.1", temperature=0)

    SYSTEM_PROMPT = """You are the company Policy Assistant.
Rules you must always follow:
- Answer policy questions using ONLY the search_policy tool's results. Never invent policy details.
- If the answer is not present in the retrieved content, say you don't know and suggest contacting HR.
- Never call raise_hr_ticket unless the user has clearly confirmed they want that action taken.
- Be concise and stick to what the retrieved content actually says.
"""

    return create_agent(llm, tools, prompt=SYSTEM_PROMPT)


# ============================================================
# 2. PAGE SETUP
# ============================================================

st.set_page_config(page_title="Policy Assistant", page_icon="\U0001F4CB")
st.title("Uptor RAG Agent \u2014 Semantic Chunking + Re-ranking")
st.caption("Ask about company policy, or ask it to raise an HR request.")

agent_executor = load_agent()


# ============================================================
# 3. CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# 4. CHAT INPUT
# ============================================================

user_question = st.chat_input("Ask a question...")

if user_question:

    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = agent_executor.invoke(
                {"messages": [{"role": "user", "content": user_question}]}
            )
            answer = result["messages"][-1].content
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
