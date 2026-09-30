from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# Parte 1 Carga de data
# 1.- Carga documento
loader = TextLoader("document.txt", encoding="utf-8")
loader2 = TextLoader("asimov.txt", encoding="utf-8")
docs = loader2.load()

# 2.- Dividir en chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)
splits = text_splitter.split_documents(docs)

# 3.- Embeddings con Ollama
embeddings = OllamaEmbeddings(
    model = "nomic-embed-text"
)

# 4.- Guardado en base de datos vectorial (Chroma)
vectorstore = Chroma.from_documents(
    documents = splits,
    embedding = embeddings
)

# Parte 2 Consulta
# 5.- Modelo (LLM)
llm = Ollama(
    model = "qwen2.5:0.5b"
)

# 6.- Cadena RAG
qa = RetrievalQA.from_chain_type(
    llm = llm,
    retriever = vectorstore.as_retriever()
)

# 7.- Pregunta
query = "¿Cuál es la primera tarea de la lista?"
query2 = "¿Qué pasó el 14 de mayo de 2061?"
result = qa.invoke({"query": query2})

print("Pregunta:")
print(query2)
print("Respuesta:")
print(result["result"])