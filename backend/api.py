# api.py - FastAPI backend for RAG system

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from sqlalchemy.orm import Session
import tempfile
import os
import re
import shutil
import uuid
from datetime import datetime
from langfuse.langchain import CallbackHandler
from langfuse import Langfuse


# Import our new database functions
from backend.database import (
    init_db, 
    get_db, 
    create_ingestion_job, 
    get_ingestion_job, 
    update_ingestion_job, 
    list_ingestion_jobs,
    IngestionStatus,
    SessionLocal
)

from backend.query import load_vectordb, get_llm, create_rag_chain
from backend.ingest import ingest_document

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

# Initialize Langfuse for observability
try:
    langfuse_handler = CallbackHandler()
    langfuse_client = Langfuse()
    print("✓ Langfuse initialized successfully")
except Exception as e:
    print(f"⚠ Langfuse initialization failed: {e}")
    langfuse_handler = None
    langfuse_client = None

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    collection_name: str  # Required collection name
    k: Optional[int] = 3  # Number of chunks to retrieve (default: 3)

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
def initialize_rag_system(collection_name: str):
    """Initialize or reload the RAG system for a specific collection"""
    global rag_chains, retrievers, vectordbs
    
    if not os.path.exists("./Vector_DB"):
        return False
    
    # Import with collection parameter
    from backend.query import load_vectordb_with_collection, get_llm, create_rag_chain
    
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
    
    # Check if Vector_DB exists and has actual collections with documents
    if os.path.exists("./Vector_DB"):
        try:
            import chromadb
            client = chromadb.PersistentClient(path="./Vector_DB")
            collections = client.list_collections()
            
            # Only initialize if there are collections with documents
            if collections:
                for col in collections:
                    if col.count() > 0:  # Only load collections with documents
                        try:
                            initialize_rag_system(col.name)
                            print(f"✓ RAG system initialized with collection: {col.name}")
                        except Exception as e:
                            print(f"⚠ Could not load collection {col.name}: {e}")
                        break  # Load first non-empty collection
            else:
                print("⚠ No collections found. Upload documents first.")
        except Exception as e:
            print(f"⚠ Could not initialize RAG system: {e}")
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
    """Query the RAG system with detailed tracing"""
    collection_name = request.collection_name
    k = request.k
    
    # Validate k value
    if k < 1 or k > 20:
        raise HTTPException(status_code=400, detail="k must be between 1 and 20")
    
    # Check if collection is loaded
    if collection_name not in vectordbs:
        if not initialize_rag_system(collection_name):
            raise HTTPException(
                status_code=503,
                detail=f"Collection '{collection_name}' not found."
            )
    
    try:
        from backend.query import get_llm, create_rag_chain
        
        vectordb = vectordbs[collection_name]
        llm = get_llm()
        rag_chain, retriever = create_rag_chain(vectordb, llm, k=k)
        
        # ========== ENHANCED TRACING ==========
        if langfuse_handler and langfuse_client:
            try:
                # Create main trace
                trace = langfuse_client.trace(
                    name="rag-query-complete",
                    input={"question": request.question, "k": k, "collection": collection_name},
                    metadata={"endpoint": "/query", "user_agent": "streamlit-ui"}
                )
                
                # Retrieval span
                retrieval_span = trace.span(
                    name="document-retrieval",
                    input={"question": request.question, "k": k},
                    metadata={"collection": collection_name}
                )
                
                source_docs = retriever.invoke(request.question)
                
                retrieval_span.end(
                    output={
                        "num_chunks_retrieved": len(source_docs),
                        "avg_chunk_length": sum(len(d.page_content) for d in source_docs) / len(source_docs) if source_docs else 0,
                        "total_context_chars": sum(len(d.page_content) for d in source_docs)
                    }
                )
                
                # Generation span
                generation_span = trace.span(
                    name="answer-generation",
                    input={
                        "question": request.question,
                        "context_chunks": len(source_docs)
                    }
                )
                
                answer = rag_chain.invoke(
                    request.question,
                    config={"callbacks": [langfuse_handler]}
                )
                
                generation_span.end(
                    output={"answer": answer, "answer_length": len(answer)},
                    metadata={"model": "gemini-2.5-flash"}
                )
                
                # Complete trace
                trace.update(
                    output={"answer": answer, "chunks_used": len(source_docs)},
                    tags=["rag", f"k-{k}", collection_name]
                )
            except Exception as trace_error:
                # If tracing fails, continue without it
                print(f"⚠ Tracing error (continuing): {trace_error}")
                answer = rag_chain.invoke(request.question)
                source_docs = retriever.invoke(request.question)
            
        else:
            # Fallback without tracing
            answer = rag_chain.invoke(request.question)
            source_docs = retriever.invoke(request.question)
        
        # Format response
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
        from backend.ingest import ingest_document_to_collection
        
        # Update progress
        update_ingestion_job(
            db,
            ingestion_id,
            message="Chunking document...",
            progress=40
        )
        
        # THE SLOW PART (2-5 minutes) - but user already got their response!
        # Strip whitespace to ensure ChromaDB compatibility
        collection_name = collection_name.strip()
        
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
@app.post("/ingest", response_model=IngestStartResponse)
async def ingest_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to ingest"),
    collection_name: str = Form(..., description="Name of the collection"),
    chunking_strategy: str = Form("semantic", description="semantic or fixed"),
    db: Session = Depends(get_db)
):
    
    # Step 1: Clean the input (remove extra spaces)
    collection_name = collection_name.strip()
    
    # Step 2: Validate length
    if len(collection_name) < 3:
        raise HTTPException(
            status_code=400,
            detail="Collection name must be at least 3 characters long"
        )
    
    if len(collection_name) > 512:
        raise HTTPException(
            status_code=400,
            detail="Collection name must be less than 512 characters"
        )
    
    # Step 3: Validate format (must start/end with letter or number)
    if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]$', collection_name):
        raise HTTPException(
            status_code=400,
            detail="Collection name must start and end with a letter or number, "
                   "and can only contain letters, numbers, dots (.), underscores (_), or hyphens (-)"
        )
    # ========== END VALIDATION BLOCK ==========
    
    # Validate file type (EXISTING CODE - DON'T CHANGE)
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Validate chunking strategy (EXISTING CODE - DON'T CHANGE)
    if chunking_strategy not in ["semantic", "fixed"]:
        raise HTTPException(status_code=400, detail="Invalid chunking_strategy")
    
    # NEW: Generate unique ingestion ID
    ingestion_id = str(uuid.uuid4())
    
    # Create temp file to save upload
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_path = temp_file.name
    temp_file.close()
    
    try:
        # Save uploaded file to temp path
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Create database record for tracking
        create_ingestion_job(
            db,
            ingestion_id=ingestion_id,
            collection_name=collection_name,
            chunking_strategy=chunking_strategy,
            original_filename=file.filename,
            status=IngestionStatus.PENDING
        )
        
        # Start background task
        background_tasks.add_task(
            process_ingestion_background,
            ingestion_id,
            temp_path,
            collection_name,
            chunking_strategy,
            file.filename
        )
        
        # Return immediately with ingestion ID
        return IngestStartResponse(
            ingestion_id=ingestion_id,
            message=f"Ingestion started for '{file.filename}'",
            status="pending"
        )
        
    except Exception as e:
        # Clean up temp file if error occurs before background task starts
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Failed to start ingestion: {str(e)}")


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


# @app.get("/ingestions") #(optional, for admin/debugging)
# async def list_all_ingestions(
#     limit: int = 50,
#     db: Session = Depends(get_db)
# ):
#     """
#     NEW ENDPOINT - List all ingestion jobs (most recent first).
    
#     Useful for:
#     - Seeing all past ingestions
#     - Debugging
#     - Admin dashboard
    
#     Example: GET /ingestions?limit=20
#     """
#     jobs = list_ingestion_jobs(db, limit=limit)
#     return {
#         "total": len(jobs),
#         "ingestions": [job.to_dict() for job in jobs]
#     }


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
