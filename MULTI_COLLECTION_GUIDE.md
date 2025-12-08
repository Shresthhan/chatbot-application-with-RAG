# Multi-Collection RAG System - User Guide

## Overview
The system now supports multiple collections, allowing you to organize different documents separately and query them independently.

## Key Features

### 🎯 Collection Management
- **Create Collections**: Automatically created when ingesting documents with a new collection name
- **Select Collections**: Choose which collection to query via dropdown in sidebar
- **View Collections**: See all collections with their chunk counts
- **Independent Context**: Each collection maintains its own document context

### 📚 Use Cases

1. **Research Papers by Topic**
   - Collection: "machine_learning" - AI/ML papers
   - Collection: "quantum_computing" - Quantum papers
   - Collection: "biology" - Biology papers

2. **Project Documentation**
   - Collection: "project_alpha_docs" - Project Alpha documentation
   - Collection: "project_beta_docs" - Project Beta documentation

3. **Language/Domain Separation**
   - Collection: "technical_docs" - Technical documentation
   - Collection: "business_docs" - Business documents

## How to Use

### Step 1: Start the API Backend
```powershell
# In terminal
.venv\Scripts\python.exe api.py
```

The API will start on http://localhost:8000
- API docs available at: http://localhost:8000/docs

### Step 2: Start Streamlit UI
```powershell
# In another terminal
.venv\Scripts\python.exe -m streamlit run app_api.py
```

### Step 3: Create Your First Collection

1. Click **Ingestion** tab in sidebar
2. Enter a collection name (e.g., "research_papers")
3. Upload a PDF file
4. Click **Ingest Document**
5. Wait for processing (2-5 minutes)

### Step 4: Query Your Collection

1. Select collection from dropdown in sidebar
2. Type your question in chat input
3. Get answers based only on that collection's documents

### Step 5: Create Additional Collections

1. Go to **Ingestion** tab
2. Enter a NEW collection name (e.g., "technical_docs")
3. Upload different PDF
4. Ingest to new collection
5. Switch between collections using dropdown

## API Endpoints

### GET /health
Health check with database status
```json
{
  "status": "healthy",
  "database_exists": true,
  "num_chunks": 150
}
```

### GET /collections
List all collections
```json
{
  "collections": [
    {"name": "my_docss", "num_chunks": 50},
    {"name": "research_papers", "num_chunks": 75},
    {"name": "technical_docs", "num_chunks": 25}
  ]
}
```

### POST /query
Query a specific collection
```json
Request:
{
  "question": "What is transformer architecture?",
  "collection_name": "research_papers"
}

Response:
{
  "answer": "...",
  "chunks": [...],
  "collection_name": "research_papers"
}
```

### POST /ingest
Ingest document to collection
```
Form Data:
- file: PDF file
- collection_name: "my_collection"

Response:
{
  "message": "Document ingested successfully",
  "num_chunks": 50,
  "collection_name": "my_collection"
}
```

### DELETE /database
Delete entire database or specific collection
```
Query Parameter:
- collection_name (optional): "specific_collection"

Response:
{
  "message": "Database cleared successfully"
}
```

## Testing

Run the test suite to verify functionality:
```powershell
.venv\Scripts\python.exe test_collections.py
```

This will:
- Check API connectivity
- List all collections
- Test querying each collection

## Architecture

```
┌─────────────────────────────────────────────────┐
│           Streamlit UI (app_api.py)             │
│  - Collection Selector Dropdown                 │
│  - Chat Interface (collection-aware)            │
│  - Document Ingestion (collection input)        │
└────────────────────┬────────────────────────────┘
                     │ HTTP Requests
                     ↓
┌─────────────────────────────────────────────────┐
│          FastAPI Backend (api.py)               │
│  - Collection-based caching                     │
│  - Multi-collection endpoints                   │
│  - Independent RAG chains per collection        │
└────────────────────┬────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────┐
│         ChromaDB (Vector_DB directory)          │
│  - Collection: my_docss                         │
│  - Collection: research_papers                  │
│  - Collection: technical_docs                   │
│  - ... (each with independent embeddings)       │
└─────────────────────────────────────────────────┘
```

## Technical Details

### Backend Changes
- **api.py**: 
  - Global state changed to dictionaries (rag_chains{}, retrievers{}, vectordbs{})
  - Collection-based caching for efficient memory usage
  - New /collections endpoint

- **query.py**: 
  - `load_vectordb_with_collection(collection_name)` function
  - Backward compatible with default collection

- **ingest.py**: 
  - `ingest_document_to_collection()` with collection parameter
  - Backward compatible with default collection

### Frontend Changes
- **app_api.py**:
  - Session state: current_collection, available_collections
  - Collection selector dropdown
  - Collection-aware query and ingestion
  - Display active collection info

## Best Practices

1. **Naming Collections**: Use descriptive, lowercase names with underscores
   - Good: "ml_research_2024", "project_docs"
   - Avoid: "Collection 1", "test", "temp"

2. **Collection Organization**: Group related documents together
   - Don't mix topics in one collection
   - Create new collections for different domains

3. **Memory Management**: Each collection loads separately
   - Only active collection uses memory
   - Switch collections freely without restarting

4. **Document Updates**: To update a collection
   - Delete old collection (if needed): DELETE /database?collection_name=...
   - Re-ingest documents with same collection name

## Troubleshooting

### Collection not showing up
- Refresh page or rerun Streamlit
- Check API logs for errors
- Verify document was ingested successfully

### Wrong collection answering
- Confirm correct collection selected in dropdown
- Check collection name in spinner during query

### API connection errors
- Ensure API server is running (port 8000)
- Check firewall settings
- Verify no port conflicts

## Future Enhancements

Potential features for future versions:
- Collection metadata (description, tags, creation date)
- Multi-collection search (query across multiple collections)
- Collection export/import
- Collection merging
- Access control per collection
- Collection statistics dashboard

---

**Version**: 2.0.0  
**Last Updated**: 2024  
**Multi-Collection Support**: Enabled ✓
