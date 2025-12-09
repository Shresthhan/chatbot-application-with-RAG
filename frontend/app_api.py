# app_api.py - Streamlit UI that uses FastAPI backend

import streamlit as st
import requests
import os

# Configuration
API_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS for softer button colors and chat input
st.markdown("""
<style>
    /* Softer primary button color */
    .stButton > button[kind="primary"] {
        background-color: #4A90E2 !important;
        border-color: #4A90E2 !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #357ABD !important;
        border-color: #357ABD !important;
    }
    
    /* Softer chat input outline color */
    [data-testid="stChatInput"] input:focus,
    [data-testid="stChatInput"] input:focus-visible,
    [data-testid="stChatInputTextArea"] textarea:focus,
    [data-testid="stChatInputTextArea"] textarea:focus-visible,
    .stChatInput input:focus,
    .stChatInput input:focus-visible,
    .stChatInput textarea:focus,
    .stChatInput textarea:focus-visible,
    input[aria-label="Ask a question..."]:focus,
    textarea[aria-label="Ask a question..."]:focus {
        border-color: #4A90E2 !important;
        box-shadow: 0 0 0 1px #4A90E2 !important;
        outline: 2px solid #4A90E2 !important;
        outline-offset: 0px !important;
    }
    
    [data-testid="stChatInput"] input,
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInputTextArea"] textarea,
    .stChatInput input,
    .stChatInput textarea {
        border-color: #d3d3d3 !important;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("Welcome Researcher!")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {"Session 1": []}
    st.session_state.current_session = "Session 1"
if "chat_chunks" not in st.session_state:
    st.session_state.chat_chunks = {"Session 1": []}
if "session_counter" not in st.session_state:
    st.session_state.session_counter = 1
if "sidebar_tab" not in st.session_state:
    st.session_state.sidebar_tab = "Chat History"
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "pending_session" not in st.session_state:
    st.session_state.pending_session = None
if "current_collection" not in st.session_state:
    st.session_state.current_collection = "my_docss"
if "available_collections" not in st.session_state:
    st.session_state.available_collections = []
if "last_error" not in st.session_state:
    st.session_state.last_error = None
if "active_ingestions" not in st.session_state:
    st.session_state.active_ingestions = []
if "retrieval_k" not in st.session_state:
    st.session_state.retrieval_k = 3  # Default k value

# Helper functions
def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def query_api(question: str, collection_name: str = "my_docss", k: int = 3):
    """Query the RAG system via API"""
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"question": question, "collection_name": collection_name, "k": k},
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise Exception(f"API query failed: {str(e)}")

def get_collections_api():
    """Get list of available collections via API"""
    try:
        response = requests.get(
            f"{API_URL}/collections",
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        return data.get("collections", [])
    except Exception as e:
        st.error(f"Failed to fetch collections: {str(e)}")
        return []
    
    
def check_ingestion_status_api(ingestion_id: str):
    """
    Calls FastAPI /status/{id} endpoint to get current status.
    Returns status data or error message.
    """
    try:
        response = requests.get(
            f"{API_URL}/status/{ingestion_id}",
            timeout=5
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to check status: {str(e)}"
        }


def ingest_pdf_api(uploaded_file, collection_name: str = "my_docss", chunking_strategy: str = "semantic"):
    """
    Ingest PDF via API - NOW RETURNS IMMEDIATELY with ingestion_id.
    OLD: Waited for completion
    NEW: Returns ingestion_id instantly
    """
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        data = {
            "collection_name": collection_name,
            "chunking_strategy": chunking_strategy
        }
        response = requests.post(
            f"{API_URL}/ingest",
            files=files,
            data=data,
            timeout=10  # Changed from 300 to 10 - now returns quickly!
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise Exception(f"API ingestion failed: {str(e)}")


# Sidebar
with st.sidebar:
    # Collection selector
    st.subheader("📚 Collection")
    
    # Fetch available collections
    collections = get_collections_api()
    if collections:
        st.session_state.available_collections = [col["name"] for col in collections]
        
        # Dropdown to select collection
        selected_collection = st.selectbox(
            "Select Collection",
            options=st.session_state.available_collections,
            index=st.session_state.available_collections.index(st.session_state.current_collection) 
                  if st.session_state.current_collection in st.session_state.available_collections 
                  else 0,
            key="collection_selector"
        )
        
        # Update current collection if changed
        if selected_collection != st.session_state.current_collection:
            st.session_state.current_collection = selected_collection
            st.rerun()
        
        # Display collection info
        current_col_info = next((col for col in collections if col["name"] == st.session_state.current_collection), None)
        if current_col_info:
            # API returns 'chunk_count' not 'num_chunks'
            chunk_count = current_col_info.get('chunk_count', 0)
            st.info(f"📄 {chunk_count} chunks")
    else:
        st.warning("No collections found. Upload a document to create one.")
    
    st.divider()
    
    # Retrieval k parameter
    st.subheader("🔍 Retrieval Settings")
    st.session_state.retrieval_k = st.slider(
        "Number of chunks to retrieve (k)",
        min_value=1,
        max_value=10,
        value=st.session_state.retrieval_k,
        step=1,
        help="Controls how many document chunks are retrieved for context. Experiment with different values to compare performance!"
    )
    st.caption(f"Currently retrieving **{st.session_state.retrieval_k}** chunks per query")
    
    st.divider()
    
    # Tab selection
    if st.button("Chat", use_container_width=True, 
                type="primary" if st.session_state.sidebar_tab == "Chat History" else "secondary"):
        st.session_state.sidebar_tab = "Chat History"
        st.rerun()
    
    if st.button("Ingestion", use_container_width=True,
                type="primary" if st.session_state.sidebar_tab == "Ingestion" else "secondary"):
        st.session_state.sidebar_tab = "Ingestion"
        st.rerun()
    
    st.divider()
    
    # CHAT HISTORY TAB
    if st.session_state.sidebar_tab == "Chat History":
        # New Chat button
        if st.button("🗊  New Chat", use_container_width=True):
            existing_numbers = []
            for session_name in st.session_state.chat_sessions.keys():
                if session_name.startswith("Session "):
                    try:
                        num = int(session_name.split(" ")[1])
                        existing_numbers.append(num)
                    except:
                        pass
            
            next_number = 1
            while next_number in existing_numbers:
                next_number += 1
            
            new_session = f"Session {next_number}"
            st.session_state.chat_sessions[new_session] = []
            st.session_state.chat_chunks[new_session] = []
            st.session_state.current_session = new_session
            st.session_state.session_counter = max(next_number, st.session_state.session_counter)
            st.rerun()
        
        st.divider()
        
        # List all chat sessions
        for session_name in st.session_state.chat_sessions.keys():
            session_messages = st.session_state.chat_sessions[session_name]
            
            if session_messages:
                preview = session_messages[0]["content"][:30] + "..." if len(session_messages[0]["content"]) > 30 else session_messages[0]["content"]
            else:
                preview = ""
            
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(
                    f"{'🟢' if session_name == st.session_state.current_session else '⚪'} {session_name}\n",
                    key=session_name,
                    use_container_width=True
                ):
                    st.session_state.current_session = session_name
                    st.rerun()
            
            with col2:
                if st.button("🗑", key=f"del_{session_name}", use_container_width=True, help="Delete session"):
                    if len(st.session_state.chat_sessions) > 1:
                        del st.session_state.chat_sessions[session_name]
                        if session_name in st.session_state.chat_chunks:
                            del st.session_state.chat_chunks[session_name]
                        if st.session_state.current_session == session_name:
                            st.session_state.current_session = list(st.session_state.chat_sessions.keys())[0]
                        st.rerun()
    
    # INGESTION TAB
        # INGESTION TAB
    elif st.session_state.sidebar_tab == "Ingestion":
        st.subheader("Upload New Document")
        
        # Collection name input
        collection_name_input = st.text_input(
            "Collection Name",
            value=st.session_state.current_collection,
            help="Enter a new collection name or use existing one"
        )
        
        # Show existing collections
        if st.session_state.available_collections:
            st.caption(f"Existing: {', '.join(st.session_state.available_collections)}")
        
        # Chunking strategy selector
        chunking_strategy = st.selectbox(
            "Chunking Strategy",
            options=["semantic", "fixed"],
            index=0,
            help="semantic: context-aware (slower) | fixed: fixed-size (faster)"
        )
        
        uploaded_file = st.file_uploader(
            "Drag and drop PDF file",
            type=["pdf"],
            help="Upload a research paper to ingest into the system",
            label_visibility="collapsed"
        )
        
        # ========== START INGESTION BUTTON (UPDATED) ==========
        if st.button("Start Ingestion", use_container_width=True, type="primary", 
                     disabled=uploaded_file is None or not collection_name_input):
            if uploaded_file and collection_name_input:
                # Strip whitespace from collection name
                collection_name_input = collection_name_input.strip()
                
                # Validate collection name
                if len(collection_name_input) < 3:
                    st.error("Collection name must be at least 3 characters long")
                elif not collection_name_input.replace("_", "").replace("-", "").replace(".", "").isalnum():
                    st.error("Collection name can only contain letters, numbers, dots, underscores, and hyphens")
                else:
                    try:
                        # Call API that now returns immediately
                        with st.spinner("Starting ingestion..."):
                            result = ingest_pdf_api(uploaded_file, collection_name_input, chunking_strategy)
                        
                        # Store ingestion info for tracking
                        from datetime import datetime
                        ingestion_id = result["ingestion_id"]
                        st.session_state.active_ingestions.append({
                            "id": ingestion_id,
                            "filename": uploaded_file.name,
                            "collection": collection_name_input,
                            "strategy": chunking_strategy,
                            "started_at": datetime.now().strftime("%H:%M:%S")
                        })
                        
                        # Show success messages
                        st.success("🎉 Ingestion Started!")
                        st.info(f"📌 Ingestion ID: `{ingestion_id}`")
                        st.info("💡 Processing in background. Check status below!")
                        
                    except Exception as e:
                        st.error(f"❌ Error starting ingestion: {str(e)}")
        
        st.divider()
        
        # ========== STATUS MONITOR SECTION (NEW) ==========
        st.subheader("📊 Active Ingestions")
        
        if not st.session_state.active_ingestions:
            st.info("No active ingestions. Upload a document above to get started!")
        
        else:
            # Show each ingestion's status
            for idx, ing in enumerate(st.session_state.active_ingestions):
                with st.expander(
                    f"📄 {ing['filename']} ({ing['collection']}) - Started: {ing['started_at']}", 
                    expanded=True
                ):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        # Check current status from API
                        status_data = check_ingestion_status_api(ing["id"])
                        
                        status = status_data.get("status", "unknown")
                        message = status_data.get("message", "No message")
                        progress = status_data.get("progress", 0)
                        
                        # Display status with appropriate styling
                        if status == "pending":
                            st.info(f"⏳ {message}")
                        
                        elif status == "processing":
                            st.warning(f"🔄 {message}")
                            # Show progress bar
                            if progress:
                                st.progress(progress / 100, text=f"{progress}% complete")
                        
                        elif status == "completed":
                            st.success(f"✅ {message}")
                            num_chunks = status_data.get("num_chunks", "?")
                            st.success(f"📦 Created {num_chunks} chunks")
                        
                        elif status == "failed":
                            st.error(f"❌ {message}")
                            error = status_data.get("error")
                            if error:
                                with st.expander("Error Details"):
                                    st.code(error)
                        
                        else:
                            st.warning(f"❓ Unknown status: {status}")
                        
                        # Show ingestion details
                        st.caption(f"ID: `{ing['id']}`")
                        st.caption(f"Strategy: {ing['strategy']}")
                    
                    with col2:
                        # Refresh button
                        if st.button("🔄", key=f"refresh_{idx}", help="Refresh status"):
                            st.rerun()
                        
                        st.write("")  # Spacing
                        
                        # Remove button (only show for completed/failed)
                        if status in ["completed", "failed", "error"]:
                            if st.button("✕", key=f"remove_{idx}", help="Remove from list"):
                                st.session_state.active_ingestions.pop(idx)
                                st.rerun()
            
            st.divider()
            
            # ========== AUTO-REFRESH OPTION (NEW) ==========
            col1, col2 = st.columns([2, 1])
            with col1:
                auto_refresh = st.checkbox(
                    "🔄 Auto-refresh every 5 seconds", 
                    value=False,
                    help="Automatically check status every 5 seconds"
                )
            with col2:
                if st.button("🗑️ Clear All", help="Remove all ingestions from list"):
                    st.session_state.active_ingestions = []
                    st.rerun()
            
            # Auto-refresh logic
            if auto_refresh:
                import time
                time.sleep(5)
                st.rerun()
    
    st.divider()

# Main content area - Chat Interface
# Check API health first
if not check_api_health():
    st.error("⚠️ FastAPI backend is not running!")
    st.info("Start the API with: `.venv\\Scripts\\python.exe api.py`")
    st.stop()

# Get API health status
try:
    health_response = requests.get(f"{API_URL}/health").json()
    
    if not health_response["database_exists"]:
        st.warning("⚠️ No vector database found. Please upload a document in the **Ingestion** tab first!")
    else:
        # Display any persistent errors
        if st.session_state.last_error:
            with st.expander("❌ Last Error Details", expanded=True):
                st.error(f"**Error Message:** {st.session_state.last_error['message']}")
                st.error(f"**Collection:** {st.session_state.last_error['collection']}")
                st.error(f"**API URL:** {st.session_state.last_error['api_url']}")
                st.code(st.session_state.last_error['traceback'], language="python")
                if st.button("Clear Error"):
                    st.session_state.last_error = None
                    st.rerun()
        
        # Get current session messages
        current_messages = st.session_state.chat_sessions[st.session_state.current_session]
        
        # Display welcome message if chat is empty
        if len(current_messages) == 0 and not st.session_state.pending_query:
            with st.chat_message("assistant"):
                st.markdown(f"👋 Ask me anything about your documents!\n\n📚 Currently using collection: **{st.session_state.current_collection}**")
        
        # Display chat history
        current_chunks = st.session_state.chat_chunks.get(st.session_state.current_session, [])
        assistant_response_count = 0
        
        for idx, message in enumerate(current_messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
            
            # Show chunks for assistant messages OUTSIDE the chat bubble
            if message["role"] == "assistant":
                if assistant_response_count < len(current_chunks) and current_chunks[assistant_response_count]:
                    chunk_data = current_chunks[assistant_response_count]
                    chunks = chunk_data if isinstance(chunk_data, list) else []
                    
                    with st.expander("📄 View Source Chunks", expanded=False):
                        for i, chunk in enumerate(chunks, 1):
                            st.markdown(f"**Chunk {i}** (*{chunk['length']} characters*)")
                            st.text_area(
                                f"Content",
                                chunk['content'],
                                height=200,
                                key=f"chunk_history_{assistant_response_count}_{i}",
                                disabled=True,
                                label_visibility="collapsed"
                            )
                            if i < len(chunks):
                                st.divider()
                assistant_response_count += 1
        
        # Process pending query if exists
        if st.session_state.pending_query and st.session_state.pending_session == st.session_state.current_session:
            query = st.session_state.pending_query
            
            # Get response from API (using current collection and k value)
            with st.spinner(f"Thinking... (Using collection: {st.session_state.current_collection}, retrieving {st.session_state.retrieval_k} chunks)"):
                try:
                    result = query_api(query, st.session_state.current_collection, st.session_state.retrieval_k)
                    response = result["answer"]
                    source_chunks = result["chunks"]
                    st.session_state.last_error = None  # Clear any previous error
                except Exception as e:
                    import traceback
                    error_details = {
                        "message": str(e),
                        "collection": st.session_state.current_collection,
                        "api_url": API_URL,
                        "traceback": traceback.format_exc()
                    }
                    st.session_state.last_error = error_details
                    st.session_state.pending_query = None
                    st.session_state.pending_session = None
                    st.rerun()
            
            # Display assistant response immediately
            with st.chat_message("assistant"):
                st.markdown(response)
            
            # Store chunks and response
            if st.session_state.current_session not in st.session_state.chat_chunks:
                st.session_state.chat_chunks[st.session_state.current_session] = []
            st.session_state.chat_chunks[st.session_state.current_session].append(source_chunks)
            
            # Add response to current session
            current_messages.append({"role": "assistant", "content": response})
            
            # Clear pending query
            st.session_state.pending_query = None
            st.session_state.pending_session = None
            st.rerun()
        
        # Chat input
        if prompt := st.chat_input("Ask a question..."):
            # Add user message to current session
            current_messages.append({"role": "user", "content": prompt})
            
            # Store as pending query
            st.session_state.pending_query = prompt
            st.session_state.pending_session = st.session_state.current_session
            
            st.rerun()

except Exception as e:
    st.error(f"Error communicating with API: {str(e)}")
