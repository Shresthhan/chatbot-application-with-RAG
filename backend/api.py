# api.py - FastAPI backend for RAG system

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import json
from typing import List, Dict, Optional, Any, AsyncGenerator
from sqlalchemy.orm import Session
import tempfile
import os
import sys
import re
import shutil
import uuid
from datetime import datetime

# Try to import Langfuse, but make it optional to avoid blocking the app
# Temporarily disabled - package version incompatibility
# To enable: pip install "langfuse>=2.40.0,<3.0"
CallbackHandler = None
Langfuse = None
LANGFUSE_AVAILABLE = False
print("⚠ Langfuse temporarily disabled - reinstall with: pip install 'langfuse>=2.40.0,<3.0'")

# Now import the evaluation functions
from experiments.evaluate_rag import run_evaluation
from experiments.evaluate_answers import run_answer_evaluation

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
from backend.tools import web_search_tool
import chromadb
import asyncio
from qdrant_client import QdrantClient
# from tavily import TavilyClient  # Commented out until we implement the tavily integration
 

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
if LANGFUSE_AVAILABLE:
    try:
        langfuse_handler = CallbackHandler()
        langfuse_client = Langfuse()
        print("✓ Langfuse initialized successfully")
    except Exception as e:
        print(f"⚠ Langfuse initialization failed: {e}")
        langfuse_handler = None
        langfuse_client = None
else:
    langfuse_handler = None
    langfuse_client = None
    print("⚠ Langfuse not available, running without observability")

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    collection_name: str  # Required collection name
    k: Optional[int] = 3  # Number of chunks to retrieve (default: 3)

class QueryResponse(BaseModel):
    answer: str
    chunks: List[Dict[str, Any]]
    trace_id: Optional[str] = None

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
    
# ========== EVALUATION MODELS ==========
class EvaluationRequest(BaseModel):
    dataset_name: str
    collection_name: str

class RetrievalEvalResponse(BaseModel):
    success: bool
    results: Dict[int, Dict[str, Any]]
    recommended_k: int

class AnswerEvalRequest(BaseModel):
    dataset_name: str
    collection_name: str
    k: int = 5

class AnswerEvalResponse(BaseModel):
    success: bool
    scores: List[Dict[str, float]]
    averages: Dict[str, float]

# NEW: For single answer evaluation (live chat evaluation)
class SingleAnswerEvalRequest(BaseModel):
    question: str
    answer: str
    expected_answer: Optional[str] = None  # Optional, for when user doesn't have ground truth
    trace_id: Optional[str] = None         # Optional, to link to original trace

class SingleAnswerEvalResponse(BaseModel):
    success: bool
    correctness: float
    completeness: float
    relevance: float
    overall: float
    explanation: Optional[str] = None


def is_greeting(query: str) -> bool:
    """Check if query is a simple greeting"""
    greetings = {"hi", "hello", "hey", "howdy", "hola", "greetings", "wassup", "yo", "morning", "afternoon", "evening"}
    words = query.lower().strip().strip("?!.").split()
    if len(words) <= 2 and any(w in greetings for w in words):
        return True
    return False



class AgentQueryRequest(BaseModel):
    question: str
    k: Optional[int] = 3

class UpdateCollectionDescriptionRequest(BaseModel):
    """Request model for updating collection description"""
    collection_name: str = Field(..., description="Name of the collection")
    description: str = Field(..., description="New description for the collection")


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
    """Initialize database on startup. RAG collections will be loaded on demand."""
    print("[STARTUP] Initializing database...")
    init_db()
    print("✓ Database initialized")

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

# Query endpoint - SOLUTION 3 INTEGRATED
@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query the RAG system with simplified Langfuse tracing"""
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
        
        # Custom explicit trace generation to get ID
        trace_id = str(uuid.uuid4())
        langfuse_callback = None
        
        if langfuse_client:
            try:
                # Create specific trace for this request
                trace = langfuse_client.trace(
                    id=trace_id,
                    name="streamlit_query",
                    input={"question": request.question, "k": k, "collection": collection_name},
                    metadata={
                        "retrieval_k": k,
                        "collection": collection_name,
                        "endpoint": "/query",
                        "model": "llama-3.1-8b-instant",
                        "provider": "groq"
                    }
                )
                # Get handler bound to this trace
                if hasattr(trace, 'get_langchain_handler'):
                    langfuse_callback = trace.get_langchain_handler()
                else:
                    print("⚠ trace object missing get_langchain_handler")
                    # Fall through to fallback
            except Exception as e:
                print(f"⚠ Langfuse trace creation failed: {e}")
                # Fallback to global handler to ensure connection isn't lost
                langfuse_callback = langfuse_handler
        
        # If we failed to get a specific callback but have a global one, use it
        if not langfuse_callback:
            langfuse_callback = langfuse_handler
            # If we fall back, the trace_id we generated won't match the one Langfuse uses
            # But at least logging will work.
        
        # Simple tracing with callback handler
        # We pass the callback to retriever too if possible, but standard retriever invoke might not take config the same way
        # depending on implementation. Let's focus on the chain invoke.
        
        source_docs = retriever.invoke(request.question)
        
        config = {
            "callbacks": [langfuse_callback] if langfuse_callback else [],
            "metadata": {
                "retrieval_k": k,
                "collection": collection_name,
                "endpoint": "streamlit_query",
                "num_chunks": len(source_docs),
                "model": "llama-3.1-8b-instant",
                "provider": "groq"
            }
        }
        
        # Update trace output if we have the object (optional but good for debugging)
        if langfuse_client and langfuse_callback:
            # We don't need to manually update trace input/output here because the handler does it,
            # BUT the handler does it for the CHAIN span. The root trace might need update.
            # actually trace.get_langchain_handler() usually attaches everything as spans under the trace.
            pass
        
        answer = rag_chain.invoke(request.question, config=config)
        
        # Format response
        chunks = []
        for i, doc in enumerate(source_docs, 1):
            chunks.append({
                "chunk_id": str(i),
                "content": doc.page_content,
                "metadata": doc.metadata,
                "length": len(doc.page_content)
            })
        
        return QueryResponse(answer=answer, chunks=chunks, trace_id=trace_id)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


# LangGraph ReAct Agent with Multi-Collection Support
@app.post("/langgraph_agent_query")
async def langgraph_agent_query(request: AgentQueryRequest):
    """
    LangGraph ReAct Agent: Automatically selects the best tool(s) from multiple collections.
    Uses ReAct framework for intelligent reasoning and tool selection.
    
    Returns:
        - answer: Final answer from the agent
        - tool_used: First tool used (for UI display)
        - tools_used: List of all tools the agent invoked
        - reasoning_steps: Step-by-step reasoning trace
        - chunks: Retrieved document chunks (if any)
    """
    try:
        from backend.langgraph_agent import get_agent
        
        # Get the agent and execute query
        agent = get_agent()
        result = agent.invoke(request.question)
        
        # Extract tools used
        tools_used_list = result.get("tools_used", [])
        tool_used = tools_used_list[0] if tools_used_list else "unknown"
        
        # Format intermediate steps as reasoning steps
        intermediate_steps = result.get("intermediate_steps", [])
        reasoning_steps = []
        for step in intermediate_steps:
            tool_name = step.get("tool", "unknown")
            reasoning_steps.append(f"Using tool: {tool_name}")
        
        return {
            "answer": result["answer"],
            "tool_used": tool_used,  # Single tool for UI
            "tools_used": tools_used_list,  # All tools for reference
            "reasoning_steps": reasoning_steps,
            "chunks": [],  # TODO: Extract chunks from tool outputs
            "intermediate_steps": intermediate_steps,  # Raw steps for debugging
            "success": True
        }
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"[ERROR] LangGraph agent failed:\n{error_detail}")
        raise HTTPException(status_code=500, detail=f"LangGraph agent query failed: {str(e)}")


# Collection Management Endpoints
class CreateCollectionRequest(BaseModel):
    """Request model for creating a collection"""
    collection_name: str = Field(..., description="Name of the collection")
    description: str = Field(..., description="Description for the collection (used in tool description)")

@app.post("/collections/create")
async def create_collection(request: CreateCollectionRequest):
    """
    Create a new Qdrant collection with metadata.
    The collection will automatically become available as a tool for the agent.
    """
    try:
        from backend.collection_manager import get_collection_manager
        
        # Extract from request body
        collection_name = request.collection_name.strip()
        description = request.description
        
        if len(collection_name) < 3:
            raise HTTPException(
                status_code=400,
                detail="Collection name must be at least 3 characters long"
            )
        
        if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]$', collection_name):
            raise HTTPException(
                status_code=400,
                detail="Collection name must start and end with a letter or number"
            )
        
        # Create collection
        manager = get_collection_manager()
        success = manager.create_collection(collection_name, description)
        
        if not success:
            raise HTTPException(status_code=400, detail="Collection already exists or creation failed")
        
        return {
            "success": True,
            "message": f"Collection '{collection_name}' created successfully",
            "collection_name": collection_name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create collection: {str(e)}")


@app.get("/collections/list")
async def list_all_collections():
    """
    List all available Qdrant collections with their metadata.
    Shows which tools are available for the agent.
    """
    try:
        from backend.collection_manager import get_collection_manager
        
        manager = get_collection_manager()
        collections = manager.get_all_collections_metadata()
        
        # Get detailed info for each collection
        collections_list = []
        for name, metadata in collections.items():
            info = manager.get_collection_info(name)
            collections_list.append({
                "name": name,
                "description": metadata.get("description", ""),
                "document_count": info.get("points_count", 0) if info else 0,
                "created_at": metadata.get("created_at"),
                "last_updated": metadata.get("last_updated")
            })
        
        return {
            "collections": collections_list,
            "total": len(collections_list)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")


@app.post("/collections/update_description")
async def update_collection_description(request: UpdateCollectionDescriptionRequest):
    """
    Update the description of an existing collection.
    This changes the tool description that the agent sees.
    """
    try:
        from backend.collection_manager import get_collection_manager
        
        manager = get_collection_manager()
        success = manager.update_collection_metadata(
            collection_name=request.collection_name,
            description=request.description
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Collection not found")
        
        return {
            "success": True,
            "message": f"Description updated for collection '{request.collection_name}'"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update description: {str(e)}")


@app.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """
    Delete a Qdrant collection and its metadata.
    WARNING: This permanently deletes all documents in the collection.
    """
    try:
        from backend.collection_manager import get_collection_manager
        
        manager = get_collection_manager()
        success = manager.delete_collection(collection_name)
        
        if not success:
            raise HTTPException(status_code=404, detail="Collection not found or deletion failed")
        
        return {
            "success": True,
            "message": f"Collection '{collection_name}' deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete collection: {str(e)}")


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

# NEW: Ingest to Qdrant (for agent knowledge base)
@app.post("/ingest_qdrant")
async def ingest_pdf_qdrant(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to ingest into Qdrant"),
    collection_name: str = Form(..., description="Name of the Qdrant collection"),
    chunking_strategy: str = Form("semantic", description="semantic or fixed"),
    db: Session = Depends(get_db)
):
    """
    Ingest a PDF into a specific Qdrant collection (for agent queries).
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Validate chunking strategy
    if chunking_strategy not in ["semantic", "fixed"]:
        raise HTTPException(status_code=400, detail="Invalid chunking_strategy")
    
    # Validate collection exists
    from backend.collection_manager import get_collection_manager
    manager = get_collection_manager()
    collections = manager.list_collections()
    if collection_name not in collections:
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' does not exist. Create it first.")
    
    # Generate unique ingestion ID
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
            collection_name=f"qdrant_{collection_name}",
            chunking_strategy=chunking_strategy,
            original_filename=file.filename,
            status=IngestionStatus.PENDING
        )
        
        # Start background task for Qdrant ingestion
        background_tasks.add_task(
            process_qdrant_ingestion_background,
            ingestion_id,
            temp_path,
            collection_name,
            chunking_strategy,
            file.filename
        )
        
        # Return immediately with ingestion ID
        return IngestStartResponse(
            ingestion_id=ingestion_id,
            message=f"Qdrant ingestion started for '{file.filename}'",
            status="pending"
        )
        
    except Exception as e:
        # Clean up temp file if error occurs before background task starts
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Failed to start Qdrant ingestion: {str(e)}")

# Background task function for Qdrant ingestion
def process_qdrant_ingestion_background(
    ingestion_id: str,
    file_path: str,
    collection_name: str,
    chunking_strategy: str,
    original_filename: str
):
    """Background task to ingest PDF into Qdrant collection"""
    print(f"\n[BACKGROUND TASK] Starting Qdrant ingestion: {ingestion_id}")
    print(f"[BACKGROUND TASK] File: {original_filename}")
    print(f"[BACKGROUND TASK] Collection: {collection_name}")
    print(f"[BACKGROUND TASK] Strategy: {chunking_strategy}")
    
    db = SessionLocal()
    try:
        # Update status to processing
        print(f"[BACKGROUND TASK] Updating status to PROCESSING...")
        update_ingestion_job(
            db, 
            ingestion_id, 
            status=IngestionStatus.PROCESSING, 
            message="Processing PDF and creating chunks...",
            progress=10
        )
        
        # Import here to avoid circular imports
        from backend.ingest import ingest_document_to_qdrant
        
        # Perform ingestion
        print(f"[BACKGROUND TASK] Starting document ingestion to collection '{collection_name}'...")
        update_ingestion_job(db, ingestion_id, progress=30, message=f"Ingesting document into Qdrant collection '{collection_name}'...")
        qdrant_vectorstore, num_chunks = ingest_document_to_qdrant(
            file_path=file_path,
            collection_name=collection_name,
            chunking_strategy=chunking_strategy
        )
        
        # Mark as completed
        print(f"[BACKGROUND TASK] Ingestion successful! {num_chunks} chunks created")
        update_ingestion_job(
            db,
            ingestion_id,
            status=IngestionStatus.COMPLETED,
            message=f"Successfully ingested {num_chunks} chunks into Qdrant",
            progress=100,
            num_chunks=num_chunks
        )
        
        print(f"✓ Qdrant ingestion {ingestion_id} completed: {num_chunks} chunks")
        
    except Exception as e:
        # Mark as failed
        error_message = str(e)
        import traceback
        traceback_str = traceback.format_exc()
        print(f"[BACKGROUND TASK] ERROR during ingestion:")
        print(f"[BACKGROUND TASK] {traceback_str}")
        
        update_ingestion_job(
            db,
            ingestion_id,
            status=IngestionStatus.FAILED,
            message=f"Ingestion failed: {error_message}",
            progress=0,
            error=traceback_str
        )
        print(f"❌ Qdrant ingestion {ingestion_id} failed: {error_message}")
    
    finally:
        # Always clean up
        db.close()
        
        # Delete temp file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass

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

# ========== CHROMADB RAG ENDPOINTS (Legacy System) ==========

# List ChromaDB collections endpoint
@app.get("/collections", response_model=CollectionsResponse)
async def list_collections():
    """List all available ChromaDB collections (for legacy RAG system)"""
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

# Delete ChromaDB database endpoint
@app.delete("/database")
async def delete_database(collection_name: Optional[str] = None):
    """Delete entire ChromaDB database or specific collection (for legacy RAG system)"""
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

# ========== EVALUATION ENDPOINTS ==========

@app.post("/evaluate/retrieval", response_model=RetrievalEvalResponse)
async def evaluate_retrieval_endpoint(request: EvaluationRequest):
    """
    Batch evaluation: Test retrieval quality across multiple k-values using a dataset.
    """
    try:
        print(f"[EVAL] Starting retrieval evaluation: dataset={request.dataset_name}")
        
        results = run_evaluation(
            dataset_name=request.dataset_name,
            collection_name=request.collection_name
        )
        
        if not results or len(results) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No results returned. Check if dataset '{request.dataset_name}' exists in Langfuse."
            )
        
        best_k = max(results.items(), key=lambda x: x[1]['average'])[0]
        print(f"[EVAL] Complete. Recommended k={best_k}")
        
        return RetrievalEvalResponse(
            success=True,
            results=results,
            recommended_k=best_k
        )
        
    except Exception as e:
        print(f"[EVAL ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.post("/evaluate/answers", response_model=AnswerEvalResponse)
async def evaluate_answers_endpoint(request: AnswerEvalRequest):
    """
    Batch evaluation: Test complete answer quality using LLM-as-judge on a dataset.
    """
    try:
        print(f"[EVAL] Starting answer evaluation: dataset={request.dataset_name}, k={request.k}")
        
        scores = run_answer_evaluation(
            dataset_name=request.dataset_name,
            collection_name=request.collection_name,
            k=request.k
        )
        
        if not scores or len(scores) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No scores returned. Check dataset and collection."
            )
        
        averages = {
            "correctness": sum(s["correctness"] for s in scores) / len(scores),
            "completeness": sum(s["completeness"] for s in scores) / len(scores),
            "relevance": sum(s["relevance"] for s in scores) / len(scores),
            "overall": sum(s["overall"] for s in scores) / len(scores)
        }
        
        print(f"[EVAL] Complete. Overall: {averages['overall']:.3f}")
        
        return AnswerEvalResponse(
            success=True,
            scores=scores,
            averages=averages
        )
        
    except Exception as e:
        print(f"[EVAL ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=f"Answer evaluation failed: {str(e)}")


@app.post("/evaluate/single", response_model=SingleAnswerEvalResponse)
async def evaluate_single_answer(request: SingleAnswerEvalRequest):
    """
    Live evaluation: Evaluate a single Q&A pair using LLM-as-judge.
    This is for evaluating individual chat responses in real-time.
    """
    try:
        print(f"[EVAL] Evaluating single answer for question: {request.question[:50]}...")
        
        # Import the evaluation function
        from experiments.evaluate_answers import evaluate_answer_quality
        
        # Evaluate the answer (expected is None for live evaluation)
        scores = evaluate_answer_quality(
            answer=request.answer,
            question=request.question,
            expected=request.expected_answer  # Will be None for live chat evaluation
        )
        
        print(f"[EVAL] Single answer evaluated. Overall: {scores['overall']:.3f}")
        
        # ========== LOG TO LANGFUSE WITH TRACE + SPANS + SCORES ==========
        if langfuse_client:
            try:
                import uuid
                from datetime import datetime
                
                # Generate unique trace ID OR use existing if provided
                trace_id = request.trace_id if request.trace_id else str(uuid.uuid4())
                
                # Step 1: Create the trace (or update existing) with generation event
                # If trace_id exists, this ADDS to it / Updates it
                if request.trace_id:
                    print(f"[EVAL] Attaching scores to existing trace: {trace_id}")
                    # If attaching to existing, we just want to ensure we have a handle to it
                    # We strictly want to log SCORES to this ID.
                else:
                    # New trace behavior
                    langfuse_client.generation(
                        id=trace_id,
                        name="live-answer-evaluation",
                        input={"question": request.question},
                        output={
                            "answer": request.answer[:500],
                            "scores": scores
                        },
                        model="cerebras/llama3.3-70b",
                        metadata={
                            "evaluation_type": "live",
                            "endpoint": "/evaluate/single",
                            "has_expected_answer": bool(request.expected_answer)
                        }
                    )
                
                # Step 2: Create a span for the evaluation process (THE COST/WORK)
                # We always want to see the JUDGE's work, even if attached to another trace
                span_id = str(uuid.uuid4())
                langfuse_client.span(
                    id=span_id,
                    trace_id=trace_id,
                    name="llm-as-judge-scoring",
                    input={"question": request.question, "answer": request.answer[:200]},
                    output=scores,
                    metadata={
                        "judge_model": "cerebras/llama3.3-70b",
                        "evaluation_mode": "live",
                        "context": "Added via evaluation button" 
                    }
                )
                
                # Step 3: Attach all 4 scores to the trace
                langfuse_client.score(
                    trace_id=trace_id,
                    name="correctness",
                    value=scores["correctness"],
                    data_type="NUMERIC",
                    comment=f"Factual accuracy: {request.question[:50]}..."
                )
                
                langfuse_client.score(
                    trace_id=trace_id,
                    name="completeness",
                    value=scores["completeness"],
                    data_type="NUMERIC",
                    comment="Answer completeness"
                )
                
                langfuse_client.score(
                    trace_id=trace_id,
                    name="relevance",
                    value=scores["relevance"],
                    data_type="NUMERIC",
                    comment="Question relevance"
                )
                
                langfuse_client.score(
                    trace_id=trace_id,
                    name="overall_quality",
                    value=scores["overall"],
                    data_type="NUMERIC",
                    comment=f"Overall: {scores['overall']:.3f}"
                )
                
                # Step 4: Flush immediately
                langfuse_client.flush()
                
                print(f"[EVAL] ✓ Scores logged to Langfuse Trace ID: {trace_id}")
                
            except Exception as lf_error:
                print(f"[EVAL] ⚠ Langfuse logging failed: {lf_error}")
                import traceback
                print(traceback.format_exc())
        
        return SingleAnswerEvalResponse(
            success=True,
            correctness=scores['correctness'],
            completeness=scores['completeness'],
            relevance=scores['relevance'],
            overall=scores['overall'],
            explanation=f"Evaluated using LLM-as-judge (Cerebras llama3.3-70b)"
        )
        
    except Exception as e:
        print(f"[EVAL ERROR] {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Single answer evaluation failed: {str(e)}"
        )



# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "RAG Chatbot API with LangGraph Multi-Collection Agent",
        "version": "3.0.0",
        "systems": {
            "langgraph_agent": "New intelligent multi-collection agent with ReAct framework",
            "chromadb_rag": "Legacy RAG system for backward compatibility"
        },
        "endpoints": {
            "agent": {
                "langgraph_agent_query": "POST /langgraph_agent_query - Query LangGraph ReAct agent"
            },
            "qdrant_collections": {
                "create_collection": "POST /collections/create - Create new Qdrant collection",
                "list_collections": "GET /collections/list - List all Qdrant collections",
                "update_description": "POST /collections/update_description - Update collection description",
                "delete_collection": "DELETE /collections/{name} - Delete Qdrant collection",
                "ingest_qdrant": "POST /ingest_qdrant - Ingest PDF to Qdrant collection"
            },
            "chromadb_rag": {
                "query": "POST /query - Query ChromaDB RAG (with collection_name)",
                "ingest": "POST /ingest - Ingest PDF to ChromaDB (returns ingestion_id)",
                "collections": "GET /collections - List ChromaDB collections",
                "delete_collection": "DELETE /database?collection_name=name - Delete ChromaDB collection",
                "delete_all": "DELETE /database - Delete entire ChromaDB"
            },
            "monitoring": {
                "health": "GET /health - Check API health",
                "status": "GET /status/{ingestion_id} - Check ingestion status"
            },
            "evaluation": {
                "evaluate_retrieval": "POST /evaluate/retrieval - Batch retrieval evaluation",
                "evaluate_answers": "POST /evaluate/answers - Batch answer evaluation",
                "evaluate_single": "POST /evaluate/single - Live single answer evaluation"
            }
        },
        "documentation": "See LANGGRAPH_AGENT_GUIDE.md for usage examples"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
