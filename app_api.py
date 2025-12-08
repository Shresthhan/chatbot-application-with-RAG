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

# Helper functions
def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def query_api(question: str, collection_name: str = "my_docss"):
    """Query the RAG system via API"""
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"question": question, "collection_name": collection_name},
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

def ingest_pdf_api(uploaded_file, collection_name: str = "my_docss", chunking_strategy: str = "semantic"):
    """Ingest PDF via API"""
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
            timeout=300
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
            help="semantic: SemanticChunker (context-aware, slower) | fixed: RecursiveCharacterTextSplitter (fixed size, faster)"
        )
        
        uploaded_file = st.file_uploader(
            "Drag and drop PDF file",
            type=["pdf"],
            help="Upload a research paper to ingest into the system",
            label_visibility="collapsed"
        )
        
        if st.button("Ingest Document", use_container_width=True, type="primary", disabled=uploaded_file is None or not collection_name_input):
            if uploaded_file and collection_name_input:
                try:
                    strategy_text = "semantic (context-aware)" if chunking_strategy == "semantic" else "fixed-size"
                    with st.spinner(f"Processing document with {strategy_text} chunking - May take 2-5 minutes..."):
                        result = ingest_pdf_api(uploaded_file, collection_name_input, chunking_strategy)
                    
                    # Show completion message
                    st.success("🎉 **INGESTION COMPLETED!**")
                    st.success(f"✅ {result['message']}")
                    st.success(f"✓ Created {result['num_chunks']} semantic chunks")
                    st.success(f"📚 Collection: {result['collection_name']}")
                    st.info("You can now switch to the Chat tab and start asking questions!")
                    
                    # Update current collection to the newly ingested one
                    st.session_state.current_collection = result['collection_name']
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
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
            
            # Get response from API (using current collection)
            with st.spinner(f"Thinking... (Using collection: {st.session_state.current_collection})"):
                try:
                    result = query_api(query, st.session_state.current_collection)
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
