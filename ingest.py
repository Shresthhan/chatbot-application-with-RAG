# ingest.py - Document ingestion script

# 1. IMPORTS - Libraries we need
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# 2. CONFIGURATION - Same settings as your LangFlow
PDF_PATH = "data/Conference_paper_pdf .pdf"
CHROMA_PATH = "./Vector_DB"
COLLECTION_NAME = "my_docss"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# 3. LOAD DOCUMENT (replaces File component)
def load_document(file_path):
    """Load PDF file and extract text"""
    print(f"Loading document from {file_path}...")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")
    return documents

# 4. SPLIT TEXT (replaces Recursive Text Splitter component)
def split_documents(documents):
    """Split documents into chunks"""
    print("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks

# 5. CREATE EMBEDDINGS (replaces Ollama Embeddings component)
def get_embeddings():
    """Initialize Ollama embeddings"""
    print("Initializing Ollama embeddings...")
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text:latest",
        base_url="http://localhost:11434"  # Your Ollama server
    )
    return embeddings

# 6. STORE IN VECTOR DB (replaces Chroma DB component)
def store_in_vectordb(chunks, embeddings):
    """Store chunks in Chroma vector database"""
    print(f"Storing chunks in Chroma at {CHROMA_PATH}...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME
    )
    print("✓ Documents stored successfully!")
    return vectordb

# 7. MAIN FUNCTION - Orchestrates everything
def main():
    """Run the complete ingestion pipeline"""
    print("=== Starting Document Ingestion ===\n")
    
    # Step by step execution
    documents = load_document(PDF_PATH)
    chunks = split_documents(documents)
    embeddings = get_embeddings()
    vectordb = store_in_vectordb(chunks, embeddings)
    
    print("\n=== Ingestion Complete ===")

# 8. RUN THE SCRIPT
if __name__ == "__main__":
    main()
