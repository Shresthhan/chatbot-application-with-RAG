# app.py - Streamlit UI for RAG Chatbot

import streamlit as st
from query import load_vectordb, get_llm, create_rag_chain

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

# Load RAG system (cached to avoid reloading)
@st.cache_resource
def load_rag_system():
    """Load and cache the RAG system"""
    vectordb = load_vectordb()
    llm = get_llm()
    rag_chain, retriever = create_rag_chain(vectordb, llm)
    return rag_chain, retriever

# Load the system
try:
    with st.spinner("Loading RAG system..."):
        rag_chain, retriever = load_rag_system()
    
    # Get current session messages
    current_messages = st.session_state.chat_sessions[st.session_state.current_session]
    
    # Display welcome message if chat is empty
    if len(current_messages) == 0:
        with st.chat_message("assistant"):
            st.markdown("👋 Ask me anything about your research paper!")
    
    # Display chat history
    for message in current_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the research paper..."):
        # Add user message to current session
        current_messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = rag_chain.invoke(prompt)
                st.markdown(response)
        
        # Add assistant response to current session
        current_messages.append({"role": "assistant", "content": response})
        st.rerun()
    
    # Sidebar
    with st.sidebar:
        st.header("💬 Chat History")
        
        # New Chat button
        if st.button("➕ New Chat", use_container_width=True):
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
        
        st.divider()
        
        
        st.divider()
        st.caption("Powered by LangChain & Streamlit")


except Exception as e:
    st.error(f"Error loading RAG system: {str(e)}")
    st.info("Make sure:")
    st.markdown("""
    1. Ollama is running on `localhost:11434`
    2. Vector database exists at `./Vector_DB`
    3. Run `python ingest.py` first to create the database
    """)
