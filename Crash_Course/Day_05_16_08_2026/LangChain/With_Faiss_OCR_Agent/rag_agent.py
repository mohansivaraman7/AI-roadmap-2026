"""
Agentic version of your existing policy RAG (create_db_with_pdf.py + rag_chat_llm.py)
---------------------------------------------------------------------------------------
What changed vs your rag_chat_llm.py:

1. Your FAISS retriever is now wrapped as a TOOL, instead of being called on every
   single question automatically. The agent decides when to use it.
2. The LLM is swapped from distilgpt2 -> a tool-calling capable model (Ollama, local).
   distilgpt2 cannot reliably decide "should I call a tool or not" - it was never
   trained for that. Any agent needs an instruction-tuned model with tool support.
3. Added a second example tool (raise_hr_ticket) so you can see how the agent
   chooses between tools, or uses both, based on the question.

SETUP (local, free, no API key):
    1. Install Ollama:        https://ollama.com/download
    2. Pull a tool-calling model:   ollama pull llama3.1
    3. pip install langchain langchain-ollama langchain-huggingface langchain-community faiss-cpu --break-system-packages

If you'd rather use OpenAI instead of Ollama, see the commented block below -
just swap that one section, everything else stays identical.
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ============================================================
# CONFIGURATION  (same as your existing scripts)
# ============================================================

PDF_FILE = "company_policy.pdf"
DB_PATH = "faiss_index"


# ============================================================
# 1. EMBEDDING MODEL  (unchanged from your script)
# ============================================================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 2. CREATE OR LOAD FAISS DATABASE  (unchanged from your script)
# ============================================================

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
    print("PDF pages:", len(documents))

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print("Number of chunks:", len(chunks))

    print("Creating FAISS database...")
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(DB_PATH)
    print("FAISS database saved!")

retriever = vector_db.as_retriever(search_kwargs={"k": 3})


# ============================================================
# 3. TOOLS  ->  things the agent is allowed to call
# ============================================================

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
    # Mock ticketing call - replace with your real ticketing system API
    ticket_id = "HR-20450"
    return f"Ticket {ticket_id} created for {employee_id}: [{request_type}] {details}"


tools = [search_policy, raise_hr_ticket]


# ============================================================
# 4. LLM  ->  must support tool calling (distilgpt2 does NOT)
# ============================================================

# ---- Option A: Ollama, local, free (used below) ----
from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.1", temperature=0)

# ---- Option B: OpenAI instead - just comment Option A out and uncomment this ----
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================================
# 5. AGENT
# ============================================================

SYSTEM_PROMPT = """You are the company Policy Assistant.
Rules you must always follow:
- Answer policy questions using ONLY the search_policy tool's results. Never invent policy details.
- If the answer is not present in the retrieved content, say you don't know and suggest contacting HR.
- Never call raise_hr_ticket unless the user has clearly confirmed they want that action taken.
- Be concise and stick to what the retrieved content actually says.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# ============================================================
# 6. CHAT LOOP  (same interface as your rag_chat_llm.py)
# ============================================================

print("\n======================================")
print("        AGENTIC RAG CHATBOT")
print("======================================")
print("Ask questions about your PDF, or ask it to raise a request.")
print("Type 'exit' to stop.\n")

while True:
    question = input("You: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    result = agent_executor.invoke({"input": question})

    print("\nAI:", result["output"])
    print("\n" + "=" * 60)
