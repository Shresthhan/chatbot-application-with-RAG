# Frontend

This folder contains the Streamlit UI that communicates with the FastAPI backend.

## Files

- **app_api.py** - Main Streamlit application
  - Collection selector
  - Chat interface
  - Document ingestion UI
  - Background job tracking
  - Source chunk display

## Features

- Multi-collection support
- Real-time ingestion status updates
- Collection management (create, switch, delete)
- Chunking strategy selector (semantic/fixed)
- Chat history per session
- Source document tracking

## Running

From project root:
```bash
streamlit run frontend\app_api.py
```

Or use: `start_frontend.bat`

## Configuration

API URL is configured in the file (default: http://localhost:8000)
