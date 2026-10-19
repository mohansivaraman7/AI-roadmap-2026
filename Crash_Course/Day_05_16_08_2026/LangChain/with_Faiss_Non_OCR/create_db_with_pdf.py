from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
# CONFIGURATION
# ============================================================

PDF_FILE = "company_policy.pdf"
DB_PATH = "faiss_index"


# ============================================================
# 1. READ PDF
# ============================================================

print("Reading PDF...")

loader = PyPDFLoader(PDF_FILE)

documents = loader.load()

print("PDF pages:", len(documents))


# ============================================================
# 2. SPLIT PDF INTO CHUNKS
# ============================================================

print("Splitting PDF into chunks...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print("Number of chunks:", len(chunks))


# ============================================================
# 3. CREATE EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 4. CREATE FAISS DATABASE
# ============================================================

print("Creating FAISS database...")

vector_db = FAISS.from_documents(
    chunks,
    embeddings
)


# ============================================================
# 5. SAVE FAISS DATABASE
# ============================================================

vector_db.save_local(DB_PATH)

print("\n======================================")
print("FAISS DATABASE CREATED")
print("======================================")

print("Vectors:", vector_db.index.ntotal)
print("Vector dimensions:", vector_db.index.d)

print("\nDatabase location:")
print(Path(DB_PATH).absolute())