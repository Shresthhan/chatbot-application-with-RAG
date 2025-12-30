# LangGraph Multi-Collection Agent Implementation Guide

## Overview

Your RAG chatbot now supports **multi-collection management** with **LangGraph-based ReAct agent** that intelligently routes queries to the most appropriate knowledge source.

### What's New

1. **Multiple Qdrant Collections** - Create specialized collections for different domains
2. **Dynamic Tool Generation** - Each collection automatically becomes a tool for the agent
3. **LangGraph ReAct Agent** - Framework-based intelligent reasoning and tool selection
4. **Collection Registry** - Metadata management for all collections
5. **Enhanced API Endpoints** - New endpoints for collection management and agent queries

---

## Architecture

```
User Query
    ↓
[LangGraph Agent]
    ├── Analyzes query intent
    ├── Examines available tools (collections + web search)
    └── Selects best tool based on 3-part descriptions
    ↓
[Tool Execution]
    ├── search_research_papers(query)
    ├── search_technical_docs(query)
    ├── search_company_policies(query)
    └── web_search(query)
    ↓
[Agent Synthesis]
    └── Generates final answer from tool results
```

---

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Key new package:
- `langgraph>=0.2.0` - LangGraph framework for agent state management

### 2. Verify Installation

```bash
python -c "import langgraph; print(f'LangGraph {langgraph.__version__} installed')"
```

---

## Usage Guide

### Step 1: Create Collections

Create specialized collections for different knowledge domains:

```bash
# Using curl (Windows PowerShell)
curl -X POST "http://localhost:8000/collections/create" `
  -F "collection_name=research_papers" `
  -F "description=Research papers on AI, machine learning, and deep learning. Use for academic questions, theoretical concepts, and research findings."

curl -X POST "http://localhost:8000/collections/create" `
  -F "collection_name=technical_docs" `
  -F "description=Technical documentation and API references. Use for implementation questions, code examples, and technical specifications."

curl -X POST "http://localhost:8000/collections/create" `
  -F "collection_name=company_policies" `
  -F "description=Company policies and HR guidelines. Use for workplace questions, policy inquiries, and procedural information."
```

**Important**: Collection names must:
- Be at least 3 characters long
- Start and end with a letter or number
- Contain only letters, numbers, dots (.), underscores (_), or hyphens (-)

### Step 2: Ingest Documents

Upload PDFs to specific collections:

```bash
# Ingest into research_papers collection
curl -X POST "http://localhost:8000/ingest_qdrant" `
  -F "file=@path/to/research_paper.pdf" `
  -F "collection_name=research_papers" `
  -F "chunking_strategy=semantic"

# Ingest into technical_docs collection
curl -X POST "http://localhost:8000/ingest_qdrant" `
  -F "file=@path/to/api_documentation.pdf" `
  -F "collection_name=technical_docs" `
  -F "chunking_strategy=semantic"
```

### Step 3: Query the Agent

Use the new LangGraph agent endpoint:

```bash
curl -X POST "http://localhost:8000/langgraph_agent_query" `
  -H "Content-Type: application/json" `
  -d '{"question": "What does the research say about transformer models?"}'
```

Response:
```json
{
  "answer": "According to the research papers...",
  "tools_used": ["search_research_papers"],
  "intermediate_steps": [
    {
      "tool": "search_research_papers",
      "input": {"query": "transformer models"},
      "output": "Found 5 relevant documents..."
    }
  ],
  "success": true
}
```

---

## API Endpoints

### Collection Management

#### Create Collection
```http
POST /collections/create
Form Data:
  - collection_name: string (required)
  - description: string (required)
```

#### List Collections
```http
GET /collections/list
Response: {
  "collections": [
    {
      "name": "research_papers",
      "description": "...",
      "document_count": 45,
      "created_at": "2025-12-29T...",
      "last_updated": "2025-12-29T..."
    }
  ],
  "total": 3
}
```

#### Update Collection Description
```http
POST /collections/update_description
Form Data:
  - collection_name: string
  - description: string
```

#### Delete Collection
```http
DELETE /collections/{collection_name}
```

### Document Ingestion

#### Ingest to Qdrant Collection
```http
POST /ingest_qdrant
Form Data:
  - file: PDF file (required)
  - collection_name: string (required)
  - chunking_strategy: "semantic" | "fixed" (default: "semantic")
```

### Agent Queries

#### LangGraph Agent Query (New)
```http
POST /langgraph_agent_query
JSON Body: {
  "question": "string",
  "k": 5 (optional, default: 3)
}
```

#### Legacy ReAct Agent Query
```http
POST /react_agent_query
JSON Body: {
  "question": "string",
  "k": 3 (optional)
}
```

---

## Tool Description Best Practices

When creating collections, use **3-part structured descriptions** for better agent routing:

### Template
```
**Purpose:** [What this collection contains]

**Use this tool for:**
- [Topic/keyword 1]
- [Topic/keyword 2]
- [Topic/keyword 3]

**Example queries:**
- "[Example question 1]"
- "[Example question 2]"
```

### Example: Research Papers Collection
```
**Purpose:** Research papers on AI, machine learning, deep learning, NLP, and computer vision.

**Use this tool for:**
- Academic questions and theoretical concepts
- Research findings and study results
- Papers on transformers, neural networks, and AI architectures
- Questions starting with "According to research..." or "What does research say..."

**Example queries:**
- "What does the research say about attention mechanisms?"
- "Latest findings on neural network optimization"
- "Research on few-shot learning techniques"
```

---

## How the Agent Works

### ReAct Loop

1. **Reasoning**: Agent analyzes the query and available tools
2. **Action**: Selects and calls the most relevant tool(s)
3. **Observation**: Receives tool output
4. **Decision**: Evaluates if more information is needed
5. **Repeat** or **Respond**: Either call another tool or generate final answer

### Tool Selection Logic

The agent uses **semantic matching** between:
- User's query
- Tool descriptions (3-part format)
- Conversation context

Example:
```
Query: "What does recent research say about transformers?"

Agent Reasoning:
- "research" keyword → High match with research_papers
- "say about" pattern → Academic inquiry
- No mention of implementation → Low match with technical_docs
Decision: Use search_research_papers tool
```

### Multiple Tool Usage

The agent can use multiple tools if needed:

```
Query: "Compare research findings with industry best practices"

Agent Flow:
1. search_research_papers("research findings")
2. search_technical_docs("industry best practices")
3. Synthesize results from both sources
```

---

## Configuration

### Collection Metadata

Collection metadata is stored in `collections_config.json`:

```json
{
  "research_papers": {
    "description": "Research papers on AI...",
    "created_at": "2025-12-29T10:00:00",
    "document_count": 45,
    "last_updated": "2025-12-29T15:30:00",
    "vector_size": 768,
    "distance": "COSINE"
  }
}
```

This file is automatically managed by the `CollectionManager` class.

### Custom Embeddings

Default: `sentence-transformers/all-mpnet-base-v2` (768 dimensions)

To change:
1. Update `backend/ingest.py` → `get_embeddings()`
2. Update `vector_size` in collection creation (must match embedding dim)

---

## Troubleshooting

### Agent Not Finding Collections

**Problem**: Agent says "No relevant tool found"

**Solutions**:
1. Check collections exist: `GET /collections/list`
2. Verify tool descriptions are clear and keyword-rich
3. Update descriptions: `POST /collections/update_description`

### Collection Creation Fails

**Problem**: "Collection already exists"

**Solutions**:
1. List existing collections first
2. Use unique names
3. Or delete existing: `DELETE /collections/{name}`

### Ingestion to Wrong Collection

**Problem**: Document uploaded to wrong collection

**Solutions**:
- Collections cannot be changed after ingestion
- Delete and re-ingest to correct collection
- Or create new ingestion with correct `collection_name`

### Agent Uses Wrong Tool

**Problem**: Agent consistently picks wrong collection

**Solutions**:
1. **Improve tool description** - Add more keywords
2. **Be specific in query** - Use terms from tool description
3. **Update description** - Clarify when to use this tool

Example fix:
```python
# Before (vague)
"This collection contains research papers"

# After (specific)
"Research papers on AI, ML, NLP, computer vision. Use for academic questions, theoretical concepts, research methodology. Keywords: transformer, neural network, attention mechanism, training, architecture."
```

---

## Example Workflow

### Complete Setup Example

```bash
# 1. Start backend
python -m uvicorn backend.api:app --reload

# 2. Create collections
curl -X POST "http://localhost:8000/collections/create" `
  -F "collection_name=ml_research" `
  -F "description=Machine learning research papers. Use for theoretical ML questions, algorithm explanations, and research findings."

# 3. Ingest documents
curl -X POST "http://localhost:8000/ingest_qdrant" `
  -F "file=@transformer_paper.pdf" `
  -F "collection_name=ml_research" `
  -F "chunking_strategy=semantic"

# 4. Query agent
curl -X POST "http://localhost:8000/langgraph_agent_query" `
  -H "Content-Type: application/json" `
  -d '{"question": "Explain the attention mechanism in transformers"}'

# 5. View response
# Agent automatically selected ml_research tool and provided answer
```

---

## Benefits of New System

| Feature | Old System | New System |
|---------|-----------|------------|
| Collections | Single collection | Multiple specialized collections |
| Tool Selection | Manual if-else logic | Automatic ReAct reasoning |
| Scalability | Add tools = update code | Add collections = auto-register tools |
| Memory | None | LangGraph state (session-based) |
| Debugging | Print statements | Structured intermediate steps |
| Flexibility | Fixed routing | Dynamic tool selection |

---

## Next Steps

1. **Create your collections** - Start with 2-3 domain-specific collections
2. **Ingest documents** - Upload PDFs to appropriate collections
3. **Test queries** - Try questions that span multiple domains
4. **Refine descriptions** - Update tool descriptions based on agent behavior
5. **Monitor usage** - Check `intermediate_steps` to see which tools are used

---

## Key Files Modified

- `backend/collection_manager.py` - NEW: Collection registry and metadata
- `backend/tools.py` - UPDATED: Dynamic tool generation
- `backend/langgraph_agent.py` - NEW: LangGraph ReAct agent
- `backend/ingest.py` - UPDATED: Collection-aware ingestion
- `backend/api.py` - UPDATED: New endpoints for collections and agent
- `requirements.txt` - UPDATED: Added `langgraph>=0.2.0`

---

## Support

For issues or questions:
1. Check tool descriptions in `/collections/list`
2. Review `intermediate_steps` in agent responses
3. Verify collection metadata in `collections_config.json`
4. Check Qdrant database in `./Qdrant_DB/`

---

**Implementation Complete!** 🚀

Your RAG system now has intelligent multi-collection routing with LangGraph ReAct framework.
