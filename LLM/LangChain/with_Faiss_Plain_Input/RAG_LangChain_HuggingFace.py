from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS

from transformers import pipeline


# --------------------------------
# 1. OUR COMPANY DATA
# --------------------------------

documents = [
    Document(
        page_content="Employees are entitled to 20 days of annual leave per year."
    ),
    Document(
        page_content="Employees are entitled to 10 days of sick leave per year."
    ),
    Document(
        page_content="Employees can work from home two days per week."
    )
]


# --------------------------------
# 2. SPLIT DOCUMENTS
# --------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_documents(documents)


# --------------------------------
# 3. CREATE EMBEDDINGS
# --------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------
# 4. STORE IN FAISS
# --------------------------------

vector_db = FAISS.from_documents(
    chunks,
    embeddings
)

vector_db.save_local("faiss_index")


# --------------------------------
# 5. CREATE RETRIEVER
# --------------------------------

retriever = vector_db.as_retriever(
    search_kwargs={"k": 2}
)


# --------------------------------
# 6. USER QUESTION
# --------------------------------

question = "How many sick leave days do employees get?"


# --------------------------------
# 7. RETRIEVE RELEVANT DATA
# --------------------------------

docs = retriever.invoke(question)

print("\nRetrieved documents:")

for doc in docs:
    print("-", doc.page_content)


# --------------------------------
# 8. LOCAL LLM
# --------------------------------

generator = pipeline(
    "text-generation",
    model="distilgpt2",
    max_new_tokens=100
)

llm = HuggingFacePipeline(
    pipeline=generator
)


# --------------------------------
# 9. GIVE RETRIEVED DATA TO LLM
# --------------------------------

context = "\n".join(
    doc.page_content
    for doc in docs
)

prompt = f"""
Answer the question using only the information below.

Information:
{context}

Question:
{question}

Answer:
"""

answer = llm.invoke(prompt)

print("\nFinal Answer:")
print(answer)
