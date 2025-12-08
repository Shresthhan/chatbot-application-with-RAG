# app.py - Streamlit UI with no FastAPI for RAG Chatbot 

import streamlit as st
import os
import tempfile

# Import all logic from other files
from backend.query import load_vectordb, get_llm, create_rag_chain
from backend.ingest import ingest_document

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
    
    /* Softer chat input outline color - override red */
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

# Title and description
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

# Load RAG system (cached to avoid reloading)
@st.cache_resource
def load_rag_system():
    """Load and cache the RAG system"""
    vectordb = load_vectordb()
    llm = get_llm()
    rag_chain, retriever = create_rag_chain(vectordb, llm)
    return rag_chain, retriever

# Sidebar (always visible)
with st.sidebar:
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
                # Find the next available session number
                existing_numbers = []
                for session_name in st.session_state.chat_sessions.keys():
                    if session_name.startswith("Session "):
                        try:
                            num = int(session_name.split(" ")[1])
                            existing_numbers.append(num)
                        except:
                            pass
                
                # Get the smallest available number starting from 1
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
                
                # Show first user message as preview, or just session name if empty
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
            
            uploaded_file = st.file_uploader(
                "Drag and drop PDF file",
                type=["pdf"],
                help="Upload a research paper to ingest into the system",
                label_visibility="collapsed"
            )
            
            if st.button("Ingest Document", use_container_width=True, type="primary", disabled=uploaded_file is None):
                if uploaded_file:
                    # Create temp file in system temp directory
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    temp_path = temp_file.name
                    temp_file.close()
                    
                    try:
                        # Save uploaded file temporarily
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Progress tracking
                        progress_placeholder = st.empty()
                        status_placeholder = st.empty()
                        
                        def update_progress(message):
                            status_placeholder.info(f"🔄 {message}")
                        
                        with st.spinner("Processing document - May take 2-5 minutes..."):
                            # Call ingestion logic from ingest.py
                            vectordb, num_chunks = ingest_document(
                                file_path=temp_path,
                                append_mode=True,  # Always append to existing database
                                progress_callback=update_progress
                            )
                        
                        # Delete temp file immediately after successful ingestion
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        
                        status_placeholder.empty()
                        st.success(f"✅ Document ingested successfully!")
                        st.success(f"✓ Created {num_chunks} semantic chunks")
                        st.balloons()
                        
                        # Clear cache to reload with new documents
                        st.cache_resource.clear()
                        st.info("🔄 Refreshing system...")
                        import time
                        time.sleep(1)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        # Clean up temp file on error
                        if os.path.exists(temp_path):
                            try:
                                os.remove(temp_path)
                            except:
                                pass
    
        st.divider()

# Main content area - Chat Interface (Always visible)
if os.path.exists("./Vector_DB"):
    # Load the system
    try:
        with st.spinner("Loading RAG system..."):
            rag_chain, retriever = load_rag_system()
        
        # Get current session messages
        current_messages = st.session_state.chat_sessions[st.session_state.current_session]
        
        # Display welcome message if chat is empty
        if len(current_messages) == 0 and not st.session_state.pending_query:
            with st.chat_message("assistant"):
                st.markdown("👋 Ask me anything about your document!")
        
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
                    chunks = chunk_data.get('chunks', []) if isinstance(chunk_data, dict) else chunk_data
                    scores = chunk_data.get('scores', []) if isinstance(chunk_data, dict) else []
                    
                    with st.expander("📄 View Source Chunks", expanded=False):
                        for i, chunk in enumerate(chunks, 1):
                            st.markdown(f"**Chunk {i}** (*{len(chunk.page_content)} characters*)")
                            # Show full chunk with text area for scrolling
                            st.text_area(
                                f"Content",
                                chunk.page_content,
                                height=200,
                                key=f"chunk_history_{assistant_response_count}_{i}",
                                disabled=True,
                                label_visibility="collapsed"
                            )
                            if i < len(chunks):
                                st.divider()
                assistant_response_count += 1
        
        # Process pending query if exists (show thinking animation in chat)
        if st.session_state.pending_query and st.session_state.pending_session == st.session_state.current_session:
            query = st.session_state.pending_query
            
            # Get response and source chunks
            with st.spinner("Thinking..."):
                response = rag_chain.invoke(query)
                # Get the same chunks the RAG chain used
                source_chunks = retriever.invoke(query)
                chunk_scores = []  # Don't show scores, they can be misleading
            
            # Store chunks and scores FIRST before displaying
            if st.session_state.current_session not in st.session_state.chat_chunks:
                st.session_state.chat_chunks[st.session_state.current_session] = []
            st.session_state.chat_chunks[st.session_state.current_session].append({
                'chunks': source_chunks,
                'scores': chunk_scores
            })
            
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
            
            # Store as pending query (will be processed on rerun with thinking animation)
            st.session_state.pending_query = prompt
            st.session_state.pending_session = st.session_state.current_session
            
            st.rerun()
    
    except Exception as e:
        st.error(f"Error loading RAG system: {str(e)}")
else:
    st.warning("⚠️ No vector database found. Please upload a document in the **Ingestion** tab first!")
