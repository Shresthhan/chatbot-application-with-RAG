# ingest.py - Document ingestion script

# 1. IMPORTS 
import warnings
warnings.filterwarnings("ignore")

from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_groq import ChatGroq
import shutil
import os
import json
import uuid

# 2. CONFIGURATION 
# Note: PDF_PATH is kept for manual ingestion via CLI (python ingest.py)
# For UI-based ingestion, use the ingest_document() function instead
PDF_PATH = "data/Conference_paper_pdf .pdf"  # Only used when running this file directly
CHROMA_PATH = "./Vector_DB"
QDRANT_PATH = "./Qdrant_DB"
QDRANT_COLLECTION = "agent_knowledge"  # Single collection for agent queries

# Global Qdrant client (singleton to avoid concurrent access issues)
_qdrant_client = None

def get_qdrant_client():
    """Get or create the shared Qdrant client instance"""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(path=QDRANT_PATH)
        print("✓ Initialized shared Qdrant client")
    return _qdrant_client

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

# 5.5. SUMMARIZE DOCUMENT
def summarize_document(chunks):
    """Generate a one-sentence summary of the document using the first few chunks"""
    try:
        from backend.query import get_llm
        llm = get_llm()
        
        # Take first 3 chunks for context (cap at 4000 chars)
        context = "\n\n".join([c.page_content for c in chunks[:3]])[:4000]
        
        prompt = f"""Analyze the following document snippets and provide a one-sentence summary (max 20 words) 
of what this document is about. Be specific (e.g., "Research paper about AI in healthcare", "Sales report for Q3 2023").

Snippets:
{context}

Summary:"""
        
        response = llm.invoke(prompt)
        summary = response.content.strip().strip('"').strip("'")
        print(f"✓ Generated summary: {summary}")
        return summary
    except Exception as e:
        print(f"⚠ Could not generate summary: {e}")
        return "No description available"

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
    
    # Generate summary for metadata
    description = summarize_document(chunks)
    collection_metadata = {"description": description}
    
    if append_mode and os.path.exists(CHROMA_PATH):
        # Load existing database and add new documents
        vectordb = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings,
            collection_name=collection_name
        )
        # Update collection metadata
        # Note: Chroma's langchain wrapper doesn't expose easy metadata update, 
        # but we can try via the underlying collection
        try:
            vectordb._collection.modify(metadata=collection_metadata)
        except:
            pass
            
        vectordb.add_documents(chunks)
        print(f"✓ Documents appended to collection '{collection_name}'!")
    else:
        # Create new database
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PATH,
            collection_name=collection_name,
            collection_metadata=collection_metadata
        )
        print(f"✓ New collection '{collection_name}' created successfully!")
    
    return vectordb

# 6.6 STORE IN QDRANT (FOR AGENT)
def store_in_qdrant(chunks, embeddings, collection_name=QDRANT_COLLECTION, update_metadata=True):
    """Store chunks in Qdrant vector database (for agent queries)
    
    Args:
        chunks: Document chunks to store
        embeddings: Embeddings instance
        collection_name: Name of the collection (default: agent_knowledge)
        update_metadata: If True, update collection metadata in registry
    
    Returns:
        QdrantVectorStore instance
    """
    print(f"Storing chunks in Qdrant collection '{collection_name}'...")
    
    # Use shared Qdrant client
    client = get_qdrant_client()
    
    # Check if collection exists
    try:
        collections = client.get_collections().collections
        collection_exists = any(col.name == collection_name for col in collections)
    except:
        collection_exists = False
    
    # Create collection if it doesn't exist
    if not collection_exists:
        print(f"Creating new Qdrant collection '{collection_name}'...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=768,  # all-mpnet-base-v2 embedding size
                distance=Distance.COSINE
            )
        )
    
    # Store documents using LangChain's Qdrant wrapper with shared client
    # First create/get the vectorstore with the shared client
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings
    )
    
    # Add documents to the vectorstore
    vectorstore.add_documents(chunks)
    
    print(f"✓ Stored {len(chunks)} chunks in Qdrant collection '{collection_name}'")
    
    # Update collection metadata if requested
    if update_metadata:
        try:
            from backend.collection_manager import get_collection_manager
            manager = get_collection_manager()
            
            # Get current document count
            collection_info = client.get_collection(collection_name)
            doc_count = collection_info.points_count
            
            # Update metadata
            manager.update_collection_metadata(
                collection_name=collection_name,
                document_count=doc_count
            )
            print(f"✓ Updated metadata for collection '{collection_name}'")
        except Exception as e:
            print(f"⚠ Could not update metadata: {e}")
    
    return vectorstore

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

def ingest_document_to_collection(file_path, collection_name, append_mode=True, progress_callback=None, chunking_strategy="semantic", store_in_qdrant_db=False):
    """Ingest a document into a specific collection
    
    Args:
        file_path: Path to the PDF file to ingest
        collection_name: Name of the collection to ingest into
        append_mode: If True, append to existing database. If False, replace database.
        progress_callback: Optional callback function to report progress (receives message string)
        chunking_strategy: Chunking strategy - "semantic" or "fixed" (default: "semantic")
        store_in_qdrant_db: If True, also store in Qdrant for agent use (default: False)
    
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
        
        # Store in ChromaDB for manual collection selection
        vectordb = store_in_vectordb(chunks, embeddings, append_mode=append_mode, collection_name=collection_name)
        
        # Optionally store in Qdrant for agent queries
        if store_in_qdrant_db:
            store_in_qdrant(chunks, embeddings)
        
        return vectordb, len(chunks)
        
    except Exception as e:
        print(f"❌ Error during ingestion: {str(e)}")
        raise

# NEW: Ingest document to Qdrant only (for agent)
def ingest_document_to_qdrant(file_path, collection_name=QDRANT_COLLECTION, progress_callback=None, chunking_strategy="semantic"):
    """Ingest a document into Qdrant database (for agent queries)
    
    Args:
        file_path: Path to the PDF file to ingest
        collection_name: Name of the Qdrant collection (default: agent_knowledge)
        progress_callback: Optional callback function to report progress (receives message string)
        chunking_strategy: Chunking strategy - "semantic" or "fixed" (default: "semantic")
    
    Returns:
        tuple: (qdrant_vectorstore, num_chunks) - The Qdrant vector store and number of chunks created
    """
    try:
        # Load document
        documents = load_document(file_path)
        
        # Initialize embeddings
        embeddings = get_embeddings()
        
        # Split using specified chunking strategy
        chunks = split_documents(documents, embeddings=embeddings, strategy=chunking_strategy)
        
        # Store in Qdrant with metadata update
        qdrant_vectorstore = store_in_qdrant(chunks, embeddings, collection_name=collection_name, update_metadata=True)
        
        return qdrant_vectorstore, len(chunks)
        
    except Exception as e:
        print(f"❌ Error during Qdrant ingestion: {str(e)}")
        raise
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
