# api.py - FastAPI backend for RAG system

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import tempfile
import os
import shutil

from query import load_vectordb, get_llm, create_rag_chain
from ingest import ingest_document

# Initialize FastAPI app
app = FastAPI(title="RAG Chatbot API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for caching (dictionary by collection name)
rag_chains = {}
retrievers = {}
vectordbs = {}

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    collection_name: str = "my_docss"  # Default collection

class QueryResponse(BaseModel):
    answer: str
    chunks: List[Dict[str, Any]]

class IngestResponse(BaseModel):
    success: bool
    message: str
    num_chunks: int
    collection_name: str

class HealthResponse(BaseModel):
    status: str
    database_exists: bool
    total_chunks: Optional[int] = None

class CollectionInfo(BaseModel):
    name: str
    chunk_count: int

class CollectionsResponse(BaseModel):
    collections: List[CollectionInfo]

# Helper function to initialize RAG system for a specific collection
def initialize_rag_system(collection_name: str = "my_docss"):
    """Initialize or reload the RAG system for a specific collection"""
    global rag_chains, retrievers, vectordbs
    
    if not os.path.exists("./Vector_DB"):
        return False
    
    # Import with collection parameter
    from query import load_vectordb_with_collection, get_llm, create_rag_chain
    
    vectordb = load_vectordb_with_collection(collection_name)
    llm = get_llm()
    rag_chain, retriever = create_rag_chain(vectordb, llm)
    
    # Cache by collection name
    vectordbs[collection_name] = vectordb
    rag_chains[collection_name] = rag_chain
    retrievers[collection_name] = retriever
    
    return True

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    if os.path.exists("./Vector_DB"):
        # Try to load default collection
        try:
            initialize_rag_system("my_docss")
            print("✓ RAG system initialized with default collection")
        except:
            print("⚠ Could not load default collection")
    else:
        print("⚠ No vector database found. Upload documents first.")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and database status"""
    db_exists = os.path.exists("./Vector_DB")
    total_chunks = None
    
    if db_exists and vectordbs:
        try:
            # Sum chunks across all loaded collections
            total_chunks = sum(vdb._collection.count() for vdb in vectordbs.values())
        except:
            pass
    
    return HealthResponse(
        status="healthy",
        database_exists=db_exists,
        total_chunks=total_chunks
    )

# Query endpoint
@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query the RAG system for a specific collection"""
    collection_name = request.collection_name
    
    # Check if collection is loaded
    if collection_name not in rag_chains or collection_name not in retrievers:
        if not initialize_rag_system(collection_name):
            raise HTTPException(
                status_code=503,
                detail=f"Collection '{collection_name}' not found. Please ingest documents first."
            )
    
    try:
        # Get answer from RAG chain for this collection
        answer = rag_chains[collection_name].invoke(request.question)
        
        # Get source chunks from this collection
        source_docs = retrievers[collection_name].invoke(request.question)
        
        # Format chunks
        chunks = []
        for i, doc in enumerate(source_docs, 1):
            chunks.append({
                "chunk_id": str(i),
                "content": doc.page_content,
                "metadata": doc.metadata,
                "length": len(doc.page_content)
            })
        
        return QueryResponse(answer=answer, chunks=chunks)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

# Ingest endpoint
@app.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(
    file: UploadFile = File(...),
    collection_name: str = Form("my_docss")
):
    """Ingest a PDF document into a specific collection"""
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Create temp file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_path = temp_file.name
    temp_file.close()
    
    try:
        # Save uploaded file
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Import with collection support
        from ingest import ingest_document_to_collection
        
        # Ingest document to specific collection
        vectordb_result, num_chunks = ingest_document_to_collection(
            file_path=temp_path,
            collection_name=collection_name,
            append_mode=True
        )
        
        # Reload RAG system for this collection
        initialize_rag_system(collection_name)
        
        return IngestResponse(
            success=True,
            message=f"Document '{file.filename}' ingested into collection '{collection_name}'",
            num_chunks=num_chunks,
            collection_name=collection_name
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

# List collections endpoint
@app.get("/collections", response_model=CollectionsResponse)
async def list_collections():
    """List all available collections"""
    if not os.path.exists("./Vector_DB"):
        return CollectionsResponse(collections=[])
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma
        import chromadb
        
        # Get all collections
        client = chromadb.PersistentClient(path="./Vector_DB")
        all_collections = client.list_collections()
        
        collections_info = []
        for col in all_collections:
            collections_info.append(CollectionInfo(
                name=col.name,
                chunk_count=col.count()
            ))
        
        return CollectionsResponse(collections=collections_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")

# Delete database endpoint
@app.delete("/database")
async def delete_database(collection_name: Optional[str] = None):
    """Delete entire database or specific collection"""
    global rag_chains, retrievers, vectordbs
    
    if not os.path.exists("./Vector_DB"):
        raise HTTPException(status_code=404, detail="Database not found")
    
    try:
        if collection_name:
            # Delete specific collection
            import chromadb
            client = chromadb.PersistentClient(path="./Vector_DB")
            client.delete_collection(collection_name)
            
            # Clear from cache
            if collection_name in rag_chains:
                del rag_chains[collection_name]
            if collection_name in retrievers:
                del retrievers[collection_name]
            if collection_name in vectordbs:
                del vectordbs[collection_name]
            
            return {"success": True, "message": f"Collection '{collection_name}' deleted successfully"}
        else:
            # Delete entire database
            shutil.rmtree("./Vector_DB")
            rag_chains.clear()
            retrievers.clear()
            vectordbs.clear()
            return {"success": True, "message": "Database deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "RAG Chatbot API with Multi-Collection Support",
        "version": "2.0.0",
        "endpoints": {
            "health": "GET /health",
            "collections": "GET /collections",
            "query": "POST /query (with collection_name)",
            "ingest": "POST /ingest (with collection_name)",
            "delete_collection": "DELETE /database?collection_name=name",
            "delete_all": "DELETE /database"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
