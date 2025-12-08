# Backend

This folder contains the FastAPI backend for the RAG chatbot system.

## Files

- **api.py** - Main FastAPI application with all endpoints
  - `/health` - Health check
  - `/collections` - List all collections
  - `/query` - Query a collection
  - `/ingest` - Ingest PDF to collection (background processing)
  - `/status/{ingestion_id}` - Check ingestion status
  - `/ingestions` - List recent ingestions
  - `/database` - Delete collections

- **database.py** - SQLAlchemy models and database operations
  - IngestionJob model for tracking background jobs
  - Database helper functions

- **ingest.py** - Document ingestion logic
  - PDF loading
  - Text chunking (semantic & fixed-size)
  - Vector database storage

- **query.py** - RAG query logic
  - Vector database loading
  - LLM initialization (Google Gemini)
  - RAG chain creation

- **ingestions.db** - SQLite database for job tracking

## Running

From project root:
```bash
python -m uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000
```

Or use: `start_backend.bat`
