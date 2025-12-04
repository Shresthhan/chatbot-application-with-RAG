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
st.title("🤖 Research Paper Q&A Chatbot")
st.markdown("Ask questions about your uploaded research paper!")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

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
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the research paper..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = rag_chain.invoke(prompt)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This chatbot uses:
        - **RAG** (Retrieval Augmented Generation)
        - **Ollama** for embeddings
        - **ChromaDB** for vector storage
        - **Google Gemini** for answers
        """)
        
        st.divider()
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
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
