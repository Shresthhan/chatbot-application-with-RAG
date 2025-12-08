# api.py - FastAPI backend for RAG system

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from sqlalchemy.orm import Session
import tempfile
import os
import shutil
import uuid
from datetime import datetime

# Import our new database functions
from database import (
    init_db, 
    get_db, 
    create_ingestion_job, 
    get_ingestion_job, 
    update_ingestion_job, 
    list_ingestion_jobs,
    IngestionStatus,
    SessionLocal
)

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
    collection_name: str  # Required collection name

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
    

class IngestStartResponse(BaseModel):
    """
    NEW MODEL - Response when ingestion starts.
    Old IngestResponse waited for completion.
    New one returns immediately with just an ID.
    """
    ingestion_id: str
    message: str
    status: str  # Will be "pending"


class StatusResponse(BaseModel):
    """
    NEW MODEL - Response for checking status.
    Tells user what's happening with their background task.
    """
    ingestion_id: str
    status: str  # pending, processing, completed, failed
    message: str
    progress: Optional[int] = None  # 0-100
    collection_name: Optional[str] = None
    num_chunks: Optional[int] = None
    error: Optional[str] = None
    started_at: str
    completed_at: Optional[str] = None


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
    """
    Initialize database AND RAG system on startup.
    
    CHANGE: Added init_db() call to create database tables.
    """
    # NEW: Initialize database first
    init_db()
    
    # Existing: Initialize RAG system
    if os.path.exists("./Vector_DB"):
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
    

def process_ingestion_background(
    ingestion_id: str,
    file_path: str,
    collection_name: str,
    chunking_strategy: str,
    original_filename: str
):
    """
    NEW FUNCTION - Runs ingestion in background (doesn't block user).
    
    This is the slow part (2-5 minutes) that used to block the /ingest endpoint.
    Now it runs separately while user can do other things.
    
    Updates database as it progresses so user can check status.
    """
    # Open NEW database connection (background tasks need their own)
    db = SessionLocal()
    
    try:
        # Update status to PROCESSING
        update_ingestion_job(
            db, 
            ingestion_id, 
            status=IngestionStatus.PROCESSING,
            message="Processing document...",
            progress=20
        )
        
        # Import ingestion function
        from ingest import ingest_document_to_collection
        
        # Update progress
        update_ingestion_job(
            db,
            ingestion_id,
            message="Chunking document...",
            progress=40
        )
        
        # THE SLOW PART (2-5 minutes) - but user already got their response!
        vectordb_result, num_chunks = ingest_document_to_collection(
            file_path=file_path,
            collection_name=collection_name,
            append_mode=True,
            chunking_strategy=chunking_strategy
        )
        
        # Mark as COMPLETED
        update_ingestion_job(
            db,
            ingestion_id,
            status=IngestionStatus.COMPLETED,
            message=f"Successfully ingested '{original_filename}'",
            progress=100,
            num_chunks=num_chunks,
            completed_at=datetime.now()
        )
        
        # Reload RAG system
        initialize_rag_system(collection_name)
        
    except Exception as e:
        # Mark as FAILED if anything goes wrong
        update_ingestion_job(
            db,
            ingestion_id,
            status=IngestionStatus.FAILED,
            message="Ingestion failed",
            error=str(e),
            completed_at=datetime.now()
        )
    
    finally:
        # Always clean up
        db.close()
        
        # Delete temp file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass


# Ingest endpoint
@app.post("/ingest", response_model=IngestStartResponse)  # Changed response model
async def ingest_pdf(
    background_tasks: BackgroundTasks,  # NEW parameter - FastAPI magic
    file: UploadFile = File(..., description="PDF file to ingest"),
    collection_name: str = Form(..., description="Name of the collection"),
    chunking_strategy: str = Form("semantic", description="semantic or fixed"),
    db: Session = Depends(get_db)  # NEW parameter - database connection
):
    """
    CHANGED BEHAVIOR:
    OLD: User uploads → waits 5 minutes → gets result
    NEW: User uploads → gets ingestion_id in 1 second → background task continues
    
    This makes the API non-blocking. User gets ticket number and can leave.
    """
    
    # Validate file type (same as before)
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Validate chunking strategy (same as before)
    if chunking_strategy not in ["semantic", "fixed"]:
        raise HTTPException(status_code=400, detail="Invalid chunking_strategy")
    
    # NEW: Generate unique ingestion ID
    ingestion_id = str(uuid.uuid4())
    
    # Create temp file (same as before)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_path = temp_file.name
    temp_file.close()
    
    try:
        # Save uploaded file (same as before)
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # NEW: Create job record in DATABASE
        create_ingestion_job(
            db,
            ingestion_id=ingestion_id,
            status=IngestionStatus.PENDING,
            message="Ingestion queued",
            progress=0,
            collection_name=collection_name,
            chunking_strategy=chunking_strategy,
            original_filename=file.filename
        )
        
        # NEW: Schedule background task (doesn't run yet, just queued)
        background_tasks.add_task(
            process_ingestion_background,
            ingestion_id=ingestion_id,
            file_path=temp_path,
            collection_name=collection_name,
            chunking_strategy=chunking_strategy,
            original_filename=file.filename
        )
        
        # NEW: Return IMMEDIATELY (don't wait for ingestion!)
        return IngestStartResponse(
            ingestion_id=ingestion_id,
            message=f"Ingestion started for '{file.filename}'",
            status="pending"
        )
        
    except Exception as e:
        # Clean up on error (same as before)
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Failed to start: {str(e)}")


@app.get("/status/{ingestion_id}", response_model=StatusResponse)
async def check_status(
    ingestion_id: str,
    db: Session = Depends(get_db)
):
    """
    NEW ENDPOINT - Check status of a background ingestion.
    
    User calls this with their ingestion_id to see:
    - Is it still processing?
    - Is it done?
    - Did it fail?
    - What's the progress percentage?
    
    Example: GET /status/a1b2c3d4-e5f6-7890-...
    """
    
    # Look up the job in database
    job = get_ingestion_job(db, ingestion_id)
    
    # If not found, return 404 error
    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Ingestion ID '{ingestion_id}' not found"
        )
    
    # Convert database object to response format and return
    return StatusResponse(**job.to_dict())


@app.get("/ingestions") #(optional, for admin/debugging)
async def list_all_ingestions(
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    NEW ENDPOINT - List all ingestion jobs (most recent first).
    
    Useful for:
    - Seeing all past ingestions
    - Debugging
    - Admin dashboard
    
    Example: GET /ingestions?limit=20
    """
    jobs = list_ingestion_jobs(db, limit=limit)
    return {
        "total": len(jobs),
        "ingestions": [job.to_dict() for job in jobs]
    }


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
# REPLACE your @app.get("/") endpoint's return statement

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "RAG Chatbot API with Multi-Collection Support",
        "version": "2.0.0",  # Keep same or bump to 2.1.0
        "endpoints": {
            "health": "GET /health",
            "collections": "GET /collections",
            "query": "POST /query (with collection_name)",
            "ingest": "POST /ingest (returns ingestion_id)",  # Changed description
            "status": "GET /status/{ingestion_id} - NEW!",     # NEW
            "ingestions": "GET /ingestions - NEW!",            # NEW
            "delete_collection": "DELETE /database?collection_name=name",
            "delete_all": "DELETE /database"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
