from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# 1. Our own data
documents = [
    "Employees are entitled to 20 days of annual leave per year.",
    "Employees get 10 days of sick leave per year.",
    "Employees can work from home two days per week."
]

# 2. Split the data into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.create_documents(documents)

# 3. Convert chunks → embeddings → store in FAISS
embeddings = OpenAIEmbeddings()

vector_db = FAISS.from_documents(
    chunks,
    embeddings
)

# 4. Create Retriever
retriever = vector_db.as_retriever(
    search_kwargs={"k": 2}
)

# 5. User asks a question
question = "How many sick leave days do employees get?"

# 6. Retrieve relevant chunks
relevant_docs = retriever.invoke(question)

# 7. Send retrieved information to LLM
context = "\n".join(doc.page_content for doc in relevant_docs)

llm = ChatOpenAI(model="gpt-4o-mini")

prompt = f"""
Answer the question using only the information below.

Information:
{context}

Question:
{question}
"""

answer = llm.invoke(prompt)

print(answer.content)