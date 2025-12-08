# Multi-Collection RAG Chatbot 🤖📚

A powerful Retrieval-Augmented Generation (RAG) chatbot with **multi-collection support**, allowing you to organize and query different document sets independently.

## ✨ Key Features

- 🗂️ **Multi-Collection Support** - Create unlimited collections for different topics/projects
- 🔍 **Intelligent Retrieval** - Semantic search using HuggingFace embeddings
- 💬 **Natural Conversations** - Powered by Google Gemini 2.5-Flash
- 📄 **Source Tracking** - View the exact chunks used to generate answers
- 🎨 **Modern UI** - Clean Streamlit interface with collection management
- 🚀 **FastAPI Backend** - RESTful API architecture for scalability
- 📊 **Semantic Chunking** - Context-aware document splitting
- 💾 **Persistent Storage** - ChromaDB vector database

## 🆕 What's New in v2.0

### Multi-Collection Architecture
- **Create separate collections** for different document sets
- **Independent context** per collection (no cross-contamination)
- **Easy switching** between collections via dropdown
- **Collection management** - create, query, list, delete

### Enhanced UI
- Collection selector in sidebar
- Collection info display (chunk counts)
- Collection-aware ingestion
- Visual feedback for active collection

### Improved API
- Collection-based endpoints
- List all collections with statistics
- Collection-specific querying
- Selective collection deletion

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd chatbot-application-with-RAG
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file with your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

### Running the Application

#### Step 1: Start FastAPI Backend
```bash
.venv\Scripts\python.exe api.py
```
API will run on http://localhost:8000

#### Step 2: Start Streamlit UI
```bash
.venv\Scripts\python.exe -m streamlit run app_api.py
```
UI will open at http://localhost:8501

#### Step 3: Create Collections & Upload Documents
1. Go to **Ingestion** tab
2. Enter a collection name (e.g., "research_papers")
3. Upload PDF file
4. Click **Ingest Document**
5. Wait 2-5 minutes for processing

#### Step 4: Query Your Documents
1. Select collection from dropdown
2. Ask questions in the chat
3. Get answers with source chunks!

## 🏗️ Architecture

```
Streamlit UI (app_api.py)
    ↓ REST API calls
FastAPI Backend (api.py)
    ↓ Collection-specific queries
ChromaDB Vector Database
    ├── Collection: my_docss
    ├── Collection: research_papers
    └── Collection: technical_docs
```

## 🔧 Technology Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit |
| **Backend** | FastAPI |
| **LLM** | Google Gemini 2.5-Flash |
| **Embeddings** | HuggingFace (all-mpnet-base-v2) |
| **Vector DB** | ChromaDB |
| **Chunking** | SemanticChunker (LangChain) |

## 🎯 Use Cases

### Research Papers
- Collection: "ml_papers" - Machine Learning research
- Collection: "quantum_papers" - Quantum Computing papers
- Collection: "bio_papers" - Biology research

### Project Documentation
- Collection: "project_alpha" - Alpha project docs
- Collection: "project_beta" - Beta project docs
- Collection: "project_gamma" - Gamma project docs

### Multi-Domain Knowledge Base
- Collection: "technical_docs" - Technical documentation
- Collection: "business_docs" - Business documents
- Collection: "legal_docs" - Legal documents

## 🔌 API Endpoints

### GET /health
Check API health and database status

### GET /collections
List all collections with chunk counts

### POST /query
Query a specific collection
```json
{
  "question": "What is a transformer?",
  "collection_name": "research_papers"
}
```

### POST /ingest
Ingest PDF to collection
```
Form Data:
- file: PDF file
- collection_name: "my_collection"
```

### DELETE /database
Delete entire database or specific collection
```
Query Parameter:
- collection_name: "specific_collection" (optional)
```

## 🎨 UI Features

- **Collection Selector** - Dropdown to choose active collection
- **Chat Interface** - Multi-session chat history
- **Source Chunks** - Expandable view of retrieved context
- **Document Upload** - Drag-and-drop PDF ingestion
- **Collection Info** - Display chunk counts per collection
- **Session Management** - Create, switch, delete chat sessions

