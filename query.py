from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

CHROMA_DIR = "./chroma_db"

# 1.- Embeddings con Ollama (necesario para cargar el store)
embeddings = OllamaEmbeddings(
    model = "nomic-embed-text"
)

# 2.- Cargar vector (con Chroma) desde el disco
vectorstore = Chroma(
    persist_directory = CHROMA_DIR,
    embedding_function = embeddings
)

# 3.- Modelo (LLM)
llm = Ollama(
    model = "qwen2.5:0.5b"
)

# 4.- Cadena RAG
qa = RetrievalQA.from_chain_type(
    llm = llm,
    retriever = vectorstore.as_retriever()
)

# 5.- Pregunta
query = "¿Qué pasó el 14 de mayo de 2061?"
result = qa.invoke({"query": query})

print("Pregunta:")
print(query)
print("Respuesta:")
print(result["result"])