from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline

from transformers import pipeline


# ============================================================
# CONFIGURATION
# ============================================================

PDF_FILE = "company_policy.pdf"
DB_PATH = "faiss_index"


# ============================================================
# 1. EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 2. CREATE OR LOAD FAISS DATABASE
# ============================================================

if Path(DB_PATH).exists():

    print("\nFAISS database found.")
    print("Loading existing database...")

    vector_db = FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

else:

    print("\nFAISS database not found.")
    print("Reading PDF...")

    # --------------------------------------------------------
    # READ PDF
    # --------------------------------------------------------

    loader = PyPDFLoader(PDF_FILE)

    documents = loader.load()

    print("PDF pages:", len(documents))


    # --------------------------------------------------------
    # SPLIT PDF
    # --------------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    print("Number of chunks:", len(chunks))


    # --------------------------------------------------------
    # CREATE FAISS
    # --------------------------------------------------------

    print("Creating FAISS database...")

    vector_db = FAISS.from_documents(
        chunks,
        embeddings
    )


    # --------------------------------------------------------
    # SAVE FAISS
    # --------------------------------------------------------

    vector_db.save_local(DB_PATH)

    print("FAISS database saved!")


# ============================================================
# 3. CREATE RETRIEVER
# ============================================================

retriever = vector_db.as_retriever(
    search_kwargs={"k": 3}
)


# ============================================================
# 4. LOAD LOCAL LLM
# ============================================================

print("\nLoading local LLM...")

generator = pipeline(
    "text-generation",
    model="distilgpt2",
    max_new_tokens=150
)

llm = HuggingFacePipeline(
    pipeline=generator
)

print("LLM loaded!")


# ============================================================
# 5. CHAT LOOP
# ============================================================

print("\n======================================")
print("        RAG CHATBOT")
print("======================================")

print("Ask questions about your PDF.")
print("Type 'exit' to stop.\n")


while True:

    question = input("You: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break


    # ========================================================
    # 6. RETRIEVE RELEVANT DOCUMENTS
    # ========================================================

    docs = retriever.invoke(question)


    # ========================================================
    # 7. BUILD CONTEXT
    # ========================================================

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )


    # ========================================================
    # 8. CREATE PROMPT
    # ========================================================

    prompt = f"""
You are a company policy assistant.

Answer the question using ONLY the information
provided in the context.

If the answer is not present in the context,
say that you don't know.

Context:
{context}

Question:
{question}

Answer:
"""


    # ========================================================
    # 9. SEND CONTEXT + QUESTION TO LLM
    # ========================================================

    answer = llm.invoke(prompt)


    # ========================================================
    # 10. DISPLAY ANSWER
    # ========================================================

    print("\nAI:", answer)

    print("\n" + "=" * 60)