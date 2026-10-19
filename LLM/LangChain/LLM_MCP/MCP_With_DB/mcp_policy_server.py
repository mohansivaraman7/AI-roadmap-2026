"""
MCP SERVER -- exposing your RAG tools over the Model Context Protocol
=======================================================================
What this is, compared to everything you've built so far:

  rag_chat_llm.py / streamlit_agentic_rag.py
      -> ONE specific LangChain agent (with ONE specific LLM: llama3.1)
         calls your tools directly, inside that one Python program.

  THIS FILE (mcp_policy_server.py)
      -> Turns search_policy and raise_hr_ticket into an MCP SERVER --
         a standalone process that exposes these tools over a standard
         protocol (MCP) so that ANY MCP-compatible client can use them:
         Claude Desktop, Claude Code, another LangChain/LangGraph agent,
         or any other tool that speaks MCP. Your tools are no longer
         locked inside one script -- they become a reusable service.

Think of it like this: your previous scripts BUILT a chef who can cook
from a specific recipe book. This file turns the recipe book itself
into a menu that any restaurant (any MCP client) can order from.

SETUP:
    pip install "mcp[cli]" requests langchain-huggingface langchain-community \
                faiss-cpu langchain-text-splitters --break-system-packages

NOTE ON MCP SDK VERSION: this script targets mcp>=2.0.0 (released July 2026),
which renamed the old FastMCP class to MCPServer and moved its import path
from mcp.server.fastmcp to mcp.server. If you installed an older mcp<2.0.0,
either upgrade (pip install --upgrade "mcp[cli]") or swap the import back to
`from mcp.server.fastmcp import FastMCP` and use `FastMCP(...)` instead of
`MCPServer(...)` below -- the @mcp.tool() decorator usage is identical either way.

THREE TOOLS ARE EXPOSED HERE, ON PURPOSE, TO SHOW A CONTRAST:
    - search_policy       -> LOCAL only (reads your own FAISS index on disk)
    - raise_hr_ticket     -> MOCK only (hardcoded fake string, no real call)
    - check_public_holiday -> Calls YOUR OWN local API (holiday_api.py),
                               backed by YOUR OWN local SQLite database
                               (holidays.db) -- built via a 3-step pipeline:
                                 1. create_holiday_db.py  -> builds holidays.db
                                 2. holiday_api.py         -> FastAPI service
                                    in front of that DB (run with:
                                    uvicorn holiday_api:app --port 8000)
                                 3. THIS file               -> MCP tool that
                                    calls that local API over HTTP
                               This replaced an earlier version that called
                               a public third-party API (date.nager.at),
                               which was unreliable/unresponsive -- now you
                               control the data and the uptime yourself.

BEFORE RUNNING THIS SERVER, MAKE SURE:
    1. python create_holiday_db.py          (one-time, builds holidays.db)
    2. uvicorn holiday_api:app --port 8000   (leave this running in its
                                               own terminal)
    then run this MCP server as usual.

RUNNING IT DIRECTLY (for quick manual testing, prints raw MCP traffic):
    mcp dev mcp_policy_server.py

CONNECTING IT TO CLAUDE DESKTOP:
    1. Open Claude Desktop's config file:
         macOS: ~/Library/Application Support/Claude/claude_desktop_config.json
         Windows: %APPDATA%\\Claude\\claude_desktop_config.json
    2. Add an entry like this (adjust the path to where this file lives):

       {
         "mcpServers": {
           "company-policy": {
             "command": "python",
             "args": ["/full/path/to/mcp_policy_server.py"]
           }
         }
       }

    3. Restart Claude Desktop. You'll see "search_policy" and
       "raise_hr_ticket" available as tools Claude can call directly
       in a normal conversation -- no LangChain agent code needed on
       the client side at all, MCP handles that wiring.
"""

from pathlib import Path

import requests
from mcp.server import MCPServer

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
# CONFIGURATION  (same PDF + FAISS setup as your other scripts)
# ============================================================

PDF_FILE = "company_policy.pdf"
DB_PATH = "faiss_index"


# ============================================================
# 1. BUILD / LOAD THE FAISS INDEX ONCE, AT STARTUP
# ============================================================
# No Streamlit here, so no @st.cache_resource -- this just runs once
# when the MCP server process starts, and stays in memory for as long
# as the server is running (which could be the entire time Claude
# Desktop is open).

print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

if Path(DB_PATH).exists():
    print("FAISS database found. Loading existing database...")
    vector_db = FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    print("FAISS database not found. Reading PDF...")
    loader = PyPDFLoader(PDF_FILE)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(DB_PATH)
    print("FAISS database saved!")

retriever = vector_db.as_retriever(search_kwargs={"k": 3})


# ============================================================
# 2. CREATE THE MCP SERVER
# ============================================================
# MCPServer is the current (v2.0.0+) name for what used to be called
# FastMCP in mcp v1.x. Same decorator-based API (@mcp.tool()) --
# only the class name and import path changed.

mcp = MCPServer("company-policy")


@mcp.tool()
def search_policy(query: str) -> str:
    """Search the company policy document and return the most relevant
    passages for the given question. Use this for any question about
    company rules, leave, WFH, expenses, IT, or compliance."""
    docs = retriever.invoke(query)
    if not docs:
        return "No matching policy content found."
    return "\n---\n".join(d.page_content for d in docs)


@mcp.tool()
def raise_hr_ticket(employee_id: str, request_type: str, details: str) -> str:
    """Raise an HR ticket/request on behalf of an employee. Only call this
    after the user has explicitly confirmed they want the action taken --
    never call this just because a policy question was asked."""
    # Mock ticketing call -- replace with your real ticketing system API
    ticket_id = "HR-20450"
    return f"Ticket {ticket_id} created for {employee_id}: [{request_type}] {details}"


@mcp.tool()
def check_public_holiday(date: str, country_code: str = "IN") -> str:
    """Check whether a given date is a public holiday in a specific country.
    date must be in YYYY-MM-DD format. country_code is a 2-letter ISO code
    (e.g. IN for India, US for United States, GB for United Kingdom).
    Useful when an employee asks whether a leave date lands on a holiday."""

    # ------------------------------------------------------------------
    # This now calls YOUR OWN local API (holiday_api.py), backed by YOUR
    # OWN local SQLite database (holidays.db) -- instead of the external
    # date.nager.at service, which was unreliable. Same 3-layer idea as
    # search_policy (DB -> retriever) but here it's DB -> REST API -> tool.
    #
    # This still counts as a "real" external call from this tool's point
    # of view -- it's a genuine HTTP request over the network -- it's just
    # pointed at a service YOU control and run yourself, instead of a
    # public third party. Make sure holiday_api.py is running first:
    #   uvicorn holiday_api:app --port 8000
    # ------------------------------------------------------------------
    url = "http://127.0.0.1:8000/holidays"
    params = {"date": date, "country_code": country_code.upper()}

    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
    except requests.RequestException as e:
        return (f"Could not reach the local holiday API: {e}. "
                f"Make sure it's running: uvicorn holiday_api:app --port 8000")

    result = response.json()

    if result["is_holiday"]:
        return f"Yes, {date} is a public holiday in {country_code.upper()}: {result['name']} ({result['local_name']})."
    return f"No, {date} is not listed as a public holiday in {country_code.upper()}."


# ============================================================
# 3. RUN THE SERVER
# ============================================================
# stdio transport: the client (Claude Desktop, `mcp dev`, etc.) launches
# this script as a subprocess and talks to it over stdin/stdout using
# the MCP protocol. This is the standard local-server transport.

if __name__ == "__main__":
    mcp.run(transport="stdio")
