from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_DIR = "./chroma_db"

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

# 4.- Guardado en base de datos vectorial (Chroma) con persistencia en disco
vectorstore = Chroma.from_documents(
    documents = splits,
    embedding = embeddings,
    persist_directory = CHROMA_DIR 
)

print(f"Ingesta completa. {len(splits)} chunks guardados en '{CHROMA_DIR}'")