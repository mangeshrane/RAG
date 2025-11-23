from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
db = Chroma(persist_directory="./qa_agent_db", embedding_function=embedding_function)

# Test a query relevant to your docs
query = "What is the configuration for timeout?"
results = db.similarity_search(query, k=3)

for doc in results:
    print(f"--- Content: {doc.page_content[:100]}...")
    print(f"--- Metadata: {doc.metadata}") # Check if headers/service names are there
