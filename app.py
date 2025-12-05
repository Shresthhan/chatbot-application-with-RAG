# app.py - Streamlit UI for RAG Chatbot 

import streamlit as st
import os

# Import all logic from other files
from query import load_vectordb, get_llm, create_rag_chain
from ingest import ingest_document

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Title and description
st.title("Welcome Researcher!")
st.markdown("How can I help you today?")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {"Session 1": []}
    st.session_state.current_session = "Session 1"
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
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Chat History", use_container_width=True, 
                        type="primary" if st.session_state.sidebar_tab == "Chat History" else "secondary"):
                st.session_state.sidebar_tab = "Chat History"
                st.rerun()
        with col2:
            if st.button("Ingestion", use_container_width=True,
                        type="primary" if st.session_state.sidebar_tab == "Ingestion" else "secondary"):
                st.session_state.sidebar_tab = "Ingestion"
                st.rerun()
        
        st.divider()
        
        # CHAT HISTORY TAB
        if st.session_state.sidebar_tab == "Chat History":
            # New Chat button
            if st.button(" + New Chat", use_container_width=True):
                st.session_state.session_counter += 1
                new_session = f"Session {st.session_state.session_counter}"
                st.session_state.chat_sessions[new_session] = []
                st.session_state.current_session = new_session
                st.rerun()
            
            st.divider()
            
            # List all chat sessions
            for session_name in st.session_state.chat_sessions.keys():
                session_messages = st.session_state.chat_sessions[session_name]
                # Show first user message as preview, or "Empty chat"
                preview = session_messages[0]["content"][:30] + "..." if session_messages else "Empty chat"
                
                col1, col2 = st.columns([5, 1])
                with col1:
                    if st.button(
                        f"{'🟢' if session_name == st.session_state.current_session else '⚪'} {session_name}\n`{preview}`",
                        key=session_name,
                        use_container_width=True
                    ):
                        st.session_state.current_session = session_name
                        st.rerun()
                
                with col2:
                    with st.popover("⋮"):
                        if st.button("Delete", key=f"del_{session_name}", use_container_width=True):
                            if len(st.session_state.chat_sessions) > 1:
                                del st.session_state.chat_sessions[session_name]
                                if st.session_state.current_session == session_name:
                                    st.session_state.current_session = list(st.session_state.chat_sessions.keys())[0]
                                st.rerun()
        
        # INGESTION TAB
        elif st.session_state.sidebar_tab == "Ingestion":
            st.subheader("Upload New Document")
            
            uploaded_file = st.file_uploader(
                "Drag and drop PDF file",
                type=["pdf"],
                help="Upload a research paper to ingest into the system"
            )
            
            if st.button("Ingest Document", use_container_width=True, type="primary", disabled=uploaded_file is None):
                if uploaded_file:
                    temp_path = f"temp_{uploaded_file.name}"
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
        st.caption("Powered by LangChain & Streamlit")

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
        for message in current_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Process pending query if exists (show thinking animation in chat)
        if st.session_state.pending_query and st.session_state.pending_session == st.session_state.current_session:
            query = st.session_state.pending_query
            
            # Show thinking message in chat
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = rag_chain.invoke(query)
                st.markdown(response)
            
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
