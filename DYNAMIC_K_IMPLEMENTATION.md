# Dynamic k Parameter Implementation

## Overview
Implemented user-controlled dynamic `k` parameter for document chunk retrieval in the RAG system. This allows users to experiment with different numbers of retrieved chunks (k=1 to 10) and compare performance for research purposes.

## Changes Made

### 1. Backend - `query.py`
**Modified Function:**
```python
def create_rag_chain(vectordb, llm, k=3):
```
- Added `k` parameter with default value of 3
- Updated docstring to document the parameter
- Retriever now uses user-specified k: `retriever = vectordb.as_retriever(search_kwargs={"k": k})`

### 2. Backend - `api.py`
**Modified Model:**
```python
class QueryRequest(BaseModel):
    question: str
    collection_name: str
    k: Optional[int] = 3  # New parameter
```

**Modified Endpoint:**
```python
@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
```
- Accepts k parameter from request (default: 3)
- Validates k is between 1 and 20
- Creates RAG chain dynamically with user-specified k value
- No longer caches retrievers (recreates with each query for flexibility)

### 3. Frontend - `app_api.py`
**New UI Component:**
- Added k slider in sidebar under "🔍 Retrieval Settings"
- Range: 1-10 chunks
- Shows current k value in caption
- Help text explains purpose for experimentation

**Session State:**
```python
if "retrieval_k" not in st.session_state:
    st.session_state.retrieval_k = 3
```

**Updated Function:**
```python
def query_api(question: str, collection_name: str = "my_docss", k: int = 3):
```
- Passes k parameter to API
- Spinner shows current k value: "Thinking... (retrieving X chunks)"

### 4. Legacy App - `app.py`
**New Function:**
```python
def get_rag_chain_with_k(k: int):
    """Create RAG chain with specified k value"""
```
- Creates fresh RAG chain with user-specified k
- Reuses cached vectordb and LLM

**UI Changes:**
- Added same k slider as frontend
- Query execution uses dynamic k value
- Spinner shows current retrieval count

## Benefits for Research

1. **Experimentation**: Easily test k=1, 3, 5, 10 without code changes
2. **Performance Comparison**: Compare answer quality vs response time
3. **Optimal k Discovery**: Find the best k for different query types
4. **Data Collection**: Gather concrete metrics for research reports
5. **User Control**: Users can adjust based on their needs (speed vs accuracy)

## Usage Examples

### Frontend (API-based)
1. Start backend: `start_backend.bat`
2. Start frontend: `start_frontend.bat`
3. Use slider in sidebar to set k (1-10)
4. Ask questions and observe differences

### Legacy App
1. Run: `start_app.bat`
2. Use slider in sidebar to set k (1-10)
3. Ask questions directly

### API Direct Testing
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is a transformer?",
    "collection_name": "research_papers",
    "k": 5
  }'
```

## Validation
- k must be between 1 and 20 (API enforces this)
- Default value: 3 (industry standard)
- Max recommended: 10 (for performance)

## Performance Notes
- Lower k (1-3): Faster, less context, might miss information
- Medium k (3-5): Balanced approach, good for most queries
- Higher k (5-10): Slower, more context, better for complex questions

## Testing Recommendations
1. Test simple factual questions at k=1, 3, 5
2. Test complex multi-part questions at k=5, 7, 10
3. Measure response time for each k value
4. Compare answer quality and completeness
5. Document findings for research report
