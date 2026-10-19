# Company Policy RAG + Agent + MCP — Student Guide

This README walks through everything built in this course, in order, so
you can rebuild and run the whole project on your own machine without
needing to piece it back together from a chat transcript.

Every script referenced here should sit in the **same folder**, alongside
`company_policy.pdf`.

---

## 0. One-time setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt --break-system-packages
```

Also install, outside of pip:
- **Ollama** (for the local LLM): https://ollama.com/download, then `ollama pull llama3.1`
- **Node.js** (only if a Claude Desktop config references an `npx`-based
  server): https://nodejs.org — install the **LTS** version

**Common mistake:** every new project folder gets its own empty `.venv`.
If a package "already installed" still isn't found, check `which python`,
`which pip`, and `which mcp` all point to the **same** `.venv/bin/` path.

---

## 1. Basic RAG pipeline (ingestion)

**File:** `create_db_with_pdf.py`

```
PDF -> PyPDFLoader -> RecursiveCharacterTextSplitter -> HuggingFaceEmbeddings -> FAISS -> save_local()
```

Run once:
```bash
python create_db_with_pdf.py
```
Produces a `faiss_index/` folder. Safe to re-run — it skips rebuilding
if `faiss_index/` already exists.

**Key concepts:**
- `chunk_size` is a *maximum*, not a fixed size — the splitter tries
  paragraph -> line -> word breaks before cutting mid-word.
- `chunk_overlap` repeats a few characters between chunks so a phrase
  spanning a boundary isn't lost. It must always be smaller than
  `chunk_size`.
- The **same** embedding model must embed both the stored chunks and
  any future query — otherwise similarity search is meaningless.

**Inspect what got stored:** `reading_faiss.py` loads the index back and
prints both the stored text and the raw 384-number vectors.

---

## 2. OCR pipeline (for scanned PDFs only)

**File:** `create_db_with_OCR_pdf.py`

Use this instead of step 1 only if `pdf_checker.py` shows a PDF has
0 extractable text and 1 full-page image per page (i.e. it's a scan).

```
PDF page -> render to image (PyMuPDF) -> OCR text (pytesseract) -> same Split/Embed/Store flow
```

Your own test PDF (`company_policy_OCR.pdf`) proved this works: OCR
misread "5 days" as "§ days" — a real, visible OCR character error,
proof that image quality affects extraction accuracy.

---

## 3. Full RAG chat loop (terminal)

**File:** `rag_chat_llm.py`

Adds retrieval + a local LLM on top of step 1:
```
question -> retriever.invoke() -> build context -> prompt -> local LLM -> print answer
```
Run: `python rag_chat_llm.py`. Type `exit` to quit.

**Known limitation:** the original version used `distilgpt2`, a small,
non-instruction-tuned model — it does not reliably follow "answer only
from context." Swapping in `llama3.1` via Ollama fixes this (see step 4).

---

## 4. Agentic RAG (tool-calling agent)

**File:** `streamlit_agentic_rag.py`

Instead of always retrieving on every question, an LLM agent **decides**
whether to call `search_policy`, `raise_hr_ticket`, or `check_public_holiday`,
or just answer directly.

```bash
streamlit run streamlit_agentic_rag.py
```
**Not** `python streamlit_agentic_rag.py` — Streamlit apps must be
launched with `streamlit run`, or `st.session_state` and the chat UI
silently fail ("missing ScriptRunContext").

Make sure Ollama is running first (`ollama serve`, or just have the
Ollama desktop app open) and `llama3.1` is pulled, or you'll get
`ConnectError: Connection refused` or `model not found`.

**API churn to be aware of:** the agent-building API has moved twice —
`langchain.agents.create_tool_calling_agent` + `AgentExecutor` (removed)
-> `langgraph.prebuilt.create_react_agent` (deprecated) ->
`langchain.agents.create_agent` (current). This script uses the current
one. If your installed `langchain` version differs, check
`pip show langchain` and adjust the import accordingly.

**temperature=0** is set deliberately — for a factual policy assistant,
you want consistent, non-creative answers, not varied phrasing.

---

## 5. Better chunking and retrieval (optional upgrade)

**File:** `streamlit_agentic_rag_semantic_rerank.py`

Two independent upgrades over step 4:

| | Before | After |
|---|---|---|
| Chunking | `RecursiveCharacterTextSplitter` — cuts by character count, no understanding of meaning | `SemanticChunker` — cuts where the topic actually shifts, based on sentence-embedding similarity |
| Retrieval | FAISS returns top 3 by similarity, used directly | FAISS returns a wider pool (8), then a **cross-encoder** re-scores each `(question, chunk)` pair together and keeps only the best 3 |

Needs a **new** `DB_PATH` (`faiss_index_semantic`) since the chunks are
structurally different from step 1's — never mix the two.

```bash
pip install langchain-experimental sentence-transformers --break-system-packages
```

---

## 6. Real external tool: holidays (SQLite -> API -> tool)

Three layers, built in order, replacing an earlier version that called
an unreliable public API (`date.nager.at`):

**6a. `create_holiday_db.py`** — builds `holidays.db`, a local SQLite
table of sample public holidays. Run once (safe to re-run — it drops
and recreates the table):
```bash
python create_holiday_db.py
```

**6b. `holiday_api.py`** — a small FastAPI service in front of that
database. Leave this running in its own terminal:
```bash
pip install fastapi uvicorn --break-system-packages
uvicorn holiday_api:app --reload --port 8000
```
Test it directly in a browser, no MCP involved at all:
`http://127.0.0.1:8000/holidays?date=2026-01-26&country_code=IN`
and the auto-generated docs at `http://127.0.0.1:8000/docs`.

**6c.** The `check_public_holiday` tool (in both `streamlit_agentic_rag.py`
and `mcp_policy_server.py`) calls this local API with `requests.get(...)`.

---

## 7. MCP server (exposing tools beyond your own app)

**File:** `mcp_policy_server.py`

Everything above runs inside **one Python program**. MCP exists for the
opposite case: when **other, separate applications** (Claude Desktop,
Claude Code, someone else's agent) need to call the same tools, without
copy-pasting your code into each one.

```bash
pip install "mcp[cli]" --break-system-packages   # quote the brackets in zsh!
mcp dev mcp_policy_server.py
```
This opens the **MCP Inspector** in your browser — click the
"Disconnected" toggle to connect, then test each tool manually.

**Connect it to Claude Desktop instead** (edit
`claude_desktop_config.json`, see the docstring at the top of
`mcp_policy_server.py` for the exact JSON) — same server file, zero code
changes, a different client just plugs in.

**Important distinction:**
- A tool defined with `@tool` (LangChain) works only inside *that one*
  Python program.
- The *same* tool defined with `@mcp.tool()` inside an MCP server can be
  called by *any* MCP client — Python-based, Node.js-based, or a
  no-code tool like n8n. The client never knows or cares what language
  the server is written in.
- If you only ever need the tools inside your own single app (like this
  course's Streamlit app), you do **not** need MCP at all — it only
  earns its complexity once a second, separate application needs the
  same tools.

**Version churn to watch for:** the MCP Python SDK renamed its main
class from `FastMCP` (in `mcp.server.fastmcp`) to `MCPServer` (in
`mcp.server`) in version 2.0.0. Run `pip show mcp` to check which you
have, and match the import in the script accordingly.

---

## Quick troubleshooting index

| Symptom | Likely cause | Fix |
|---|---|---|
| `Error: typer is required` | `mcp[cli]` extra not installed in the **active** venv | `pip install "mcp[cli]" --break-system-packages` (quote the brackets in zsh) |
| Still fails after installing | Wrong venv active | Check `which python`/`which pip`/`which mcp` all match |
| `ModuleNotFoundError: mcp.server.fastmcp` | You have `mcp>=2.0.0`, which renamed `FastMCP` to `MCPServer` | `from mcp.server import MCPServer` |
| `ImportError: create_tool_calling_agent` | LangChain removed the old agent API | Use `from langchain.agents import create_agent` |
| `ConnectError: Connection refused` (Ollama) | Ollama server isn't running | Run `ollama serve` or open the Ollama app |
| `model 'llama3.1' not found` | Model not pulled yet | `ollama pull llama3.1` |
| `npx not found` | A different, Node.js-based MCP server is configured | Install Node.js (`brew install node` / nodejs.org installer) |
| Streamlit shows `missing ScriptRunContext` | Ran with `python` instead of `streamlit run` | `streamlit run <file>.py` |
| MCP server "just hangs", no output | Ran `python mcp_policy_server.py` directly with no client attached | Use `mcp dev mcp_policy_server.py`, or connect via Claude Desktop |

---

## Full file list

| File | Purpose |
|---|---|
| `company_policy.pdf` | Source document (clean, non-OCR) |
| `company_policy_OCR.pdf` | Source document (scanned, needs OCR) |
| `pdf_checker.py` | Detects OCR vs non-OCR PDFs |
| `create_db_with_pdf.py` | Basic ingestion (non-OCR) |
| `create_db_with_OCR_pdf.py` | Ingestion via OCR |
| `create_db_with_hardcoded.py` | Ingestion + retrieval demo, no PDF, no LLM |
| `reading_faiss.py` | Inspect a saved FAISS index |
| `rag_chat_llm.py` | Terminal RAG chat loop |
| `streamlit_agentic_rag.py` | Streamlit agentic RAG (main app) |
| `streamlit_agentic_rag_semantic_rerank.py` | Same, with semantic chunking + re-ranking |
| `create_holiday_db.py` | Builds `holidays.db` |
| `holiday_api.py` | FastAPI service over `holidays.db` |
| `mcp_policy_server.py` | MCP server exposing all tools |
| `requirements.txt` | One-command install for a fresh venv |
