# ingest.py - Document ingestion script

# 1. IMPORTS 
import warnings
warnings.filterwarnings("ignore")

from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import shutil
import os

# 2. CONFIGURATION 
# Note: PDF_PATH is kept for manual ingestion via CLI (python ingest.py)
# For UI-based ingestion, use the ingest_document() function instead
PDF_PATH = "data/Conference_paper_pdf .pdf"  # Only used when running this file directly
CHROMA_PATH = "./Vector_DB"

# 3. LOAD DOCUMENT 
def load_document(file_path):
    """Load PDF file and extract text"""
    print(f"Loading document from {file_path}...")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")
    return documents

# 4. SPLIT TEXT 
def split_documents(documents, embeddings=None, strategy="semantic"):
    """Split documents into chunks using specified strategy
    
    Args:
        documents: List of documents to split
        embeddings: Pre-initialized embeddings instance (optional, will create if not provided)
        strategy: Chunking strategy - "semantic" or "fixed"
    """
    print("Splitting document into chunks...")
    
    if strategy == "semantic":
        print("Using semantic chunking (context-aware)...")
        
        if embeddings is None:
            embeddings = get_embeddings()
        
        text_splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=None
        )
    else:  # fixed strategy
        print("Using fixed-size chunking (RecursiveCharacterTextSplitter)...")
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False
        )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks

# 5. CREATE EMBEDDINGS 
def get_embeddings():
    """Initialize HuggingFace embeddings"""
    print("Initializing HuggingFace embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True} 
    )
    return embeddings

# 6. STORE IN VECTOR DB 
def store_in_vectordb(chunks, embeddings, append_mode=False, collection_name="default"):
    """Store chunks in Chroma vector database
    
    Args:
        chunks: Document chunks to store
        embeddings: Embeddings instance
        append_mode: If True, append to existing database. If False, create new database.
        collection_name: Name of the collection (required)
    """
    
    print(f"Storing chunks in collection '{collection_name}' at {CHROMA_PATH}...")
    
    if append_mode and os.path.exists(CHROMA_PATH):
        # Load existing database and add new documents
        vectordb = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings,
            collection_name=collection_name
        )
        vectordb.add_documents(chunks)
        print(f"✓ Documents appended to collection '{collection_name}'!")
    else:
        # Create new database
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PATH,
            collection_name=collection_name
        )
        print(f"✓ New collection '{collection_name}' created successfully!")
    
    return vectordb

# 6.5. INGEST DOCUMENT (FOR UI)
def ingest_document(file_path, collection_name, append_mode=True, progress_callback=None):
    """Complete ingestion pipeline for a single document
    
    Args:
        file_path: Path to the PDF file to ingest
        collection_name: Name of the collection (required)
        append_mode: If True, append to existing database. If False, replace database.
        progress_callback: Optional callback function to report progress (receives message string)
    
    Returns:
        tuple: (vectordb, num_chunks) - The vector database and number of chunks created
    """
    return ingest_document_to_collection(file_path, collection_name, append_mode, progress_callback)

def ingest_document_to_collection(file_path, collection_name, append_mode=True, progress_callback=None, chunking_strategy="semantic"):
    """Ingest a document into a specific collection
    
    Args:
        file_path: Path to the PDF file to ingest
        collection_name: Name of the collection to ingest into
        append_mode: If True, append to existing database. If False, replace database.
        progress_callback: Optional callback function to report progress (receives message string)
        chunking_strategy: Chunking strategy - "semantic" or "fixed" (default: "semantic")
    
    Returns:
        tuple: (vectordb, num_chunks) - The vector database and number of chunks created
    """
    try:
        # Load document
        documents = load_document(file_path)
        
        # Initialize embeddings
        embeddings = get_embeddings()
        
        # Split using specified chunking strategy (this is the slow part)
        chunks = split_documents(documents, embeddings=embeddings, strategy=chunking_strategy)
        
        # Store in vector database with specific collection
        vectordb = store_in_vectordb(chunks, embeddings, append_mode=append_mode, collection_name=collection_name)
        
        return vectordb, len(chunks)
        
    except Exception as e:
        print(f"❌ Error during ingestion: {str(e)}")
        raise

# 7. MAIN FUNCTION 
def main():
    """Run the complete ingestion pipeline"""
    print("=== Starting Document Ingestion ===\n")
    
    try:
        # Clear existing vector database if it exists
        if os.path.exists(CHROMA_PATH):
            print(f"Removing existing vector database at {CHROMA_PATH}...")
            shutil.rmtree(CHROMA_PATH)
            print("✓ Old database cleared\n")
        
        # Step by step execution
        documents = load_document(PDF_PATH)
        embeddings = get_embeddings()  # Initialize once and reuse
        chunks = split_documents(documents, embeddings=embeddings)
        vectordb = store_in_vectordb(chunks, embeddings)
        
        print("\n=== Ingestion Complete ===")
        print(f"✓ Total chunks created: {len(chunks)}")
        print(f"✓ Database location: {CHROMA_PATH}")
        
    except Exception as e:
        print(f"\n❌ Error during ingestion: {str(e)}")
        raise

# 8. RUN THE SCRIPT
if __name__ == "__main__":
    main()
