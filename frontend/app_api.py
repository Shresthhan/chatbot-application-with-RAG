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
    st.session_state.current_collection = None  
if "available_collections" not in st.session_state:
    st.session_state.available_collections = []
if "last_error" not in st.session_state:
    st.session_state.last_error = None
if "active_ingestions" not in st.session_state:
    st.session_state.active_ingestions = []
if "retrieval_k" not in st.session_state:
    st.session_state.retrieval_k = 3  
if "last_query_k" not in st.session_state:
    st.session_state.last_query_k = None
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None
if "eval_running" not in st.session_state:
    st.session_state.eval_running = False
if "answer_scores" not in st.session_state:
    st.session_state.answer_scores = {}

# Helper functions
def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def query_api(question: str, collection_name: str, k: int = 3):
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
    """Calls FastAPI /status/{id} endpoint to get current status"""
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

def ingest_pdf_api(uploaded_file, collection_name: str, chunking_strategy: str = "semantic"):
    """Ingest PDF via API - returns ingestion_id instantly"""
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
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise Exception(f"API ingestion failed: {str(e)}")

def evaluate_retrieval_api(dataset_name: str, collection_name: str):
    """Call retrieval evaluation endpoint"""
    try:
        response = requests.post(
            f"{API_URL}/evaluate/retrieval",
            json={
                "dataset_name": dataset_name,
                "collection_name": collection_name
            },
            timeout=300
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise Exception(f"Retrieval evaluation failed: {str(e)}")

def evaluate_answers_api(dataset_name: str, collection_name: str, k: int, progress_callback=None):
    """Call answer evaluation endpoint with optional progress callback"""
    try:
        # If callback provided, show we're starting
        if progress_callback:
            progress_callback(0, "Starting evaluation...")
        
        response = requests.post(
            f"{API_URL}/evaluate/answers",
            json={
                "dataset_name": dataset_name,
                "collection_name": collection_name,
                "k": k
            },
            timeout=600,
            stream=False
        )
        
        if progress_callback:
            progress_callback(100, "Complete!")
            
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise Exception(f"Answer evaluation failed: {str(e)}")

def evaluate_single_answer_api(question: str, answer: str):
    """Evaluate a single Q&A pair using LLM-as-judge"""
    try:
        response = requests.post(
            f"{API_URL}/evaluate/single",
            json={
                "question": question,
                "answer": answer
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise Exception(f"Single answer evaluation failed: {str(e)}")

# Sidebar
with st.sidebar:
    # Collection selector
    st.subheader("📚 Collection")
    
    # Fetch available collections
    collections = get_collections_api()
    if collections:
        st.session_state.available_collections = [col["name"] for col in collections]
        
        if st.session_state.current_collection is None or st.session_state.current_collection not in st.session_state.available_collections:
            st.session_state.current_collection = st.session_state.available_collections[0]
        
        selected_collection = st.selectbox(
            "Select Collection",
            options=st.session_state.available_collections,
            index=st.session_state.available_collections.index(st.session_state.current_collection),
            key="collection_selector"
        )
        
        if selected_collection != st.session_state.current_collection:
            st.session_state.current_collection = selected_collection
            st.rerun()
        
        current_col_info = next((col for col in collections if col["name"] == st.session_state.current_collection), None)
        if current_col_info:
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
        help="Controls how many document chunks are retrieved for context"
    )
    st.caption(f"Currently retrieving **{st.session_state.retrieval_k}** chunks per query")
    
    st.divider()
    
    # Langfuse Observability Section
    st.subheader("🔍 Observability")
    
    langfuse_host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(
            f'<a href="{langfuse_host}" target="_blank" style="text-decoration: none; '
            f'display: block; text-align: center; padding: 10px; background-color: #4A90E2; '
            f'color: white; border-radius: 4px; font-weight: 500;">📊 Open Langfuse</a>',
            unsafe_allow_html=True
        )
    
    with col2:
        try:
            response = requests.get(f"{langfuse_host}/api/public/health", timeout=2)
            if response.status_code == 200:
                st.markdown("🟢", help="Langfuse is running")
            else:
                st.markdown("🟡", help="Langfuse status unknown")
        except:
            st.markdown("🔴", help="Langfuse is not reachable")
    
    st.caption("Monitor query performance and debug issues")
    
    if st.session_state.last_query_k is not None:
        st.info(f"📍 Last query: k={st.session_state.last_query_k}, collection='{st.session_state.current_collection}'")
    
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
    
    if st.button("Evaluation", use_container_width=True,
                type="primary" if st.session_state.sidebar_tab == "Evaluation" else "secondary"):
        st.session_state.sidebar_tab = "Evaluation"
        st.rerun()
    
    st.divider()
    
    # CHAT HISTORY TAB (in sidebar)
    if st.session_state.sidebar_tab == "Chat History":
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
    
    # INGESTION TAB (in sidebar)
    elif st.session_state.sidebar_tab == "Ingestion":
        st.subheader("Upload New Document")
        
        collection_name_input = st.text_input(
            "Collection Name",
            value=st.session_state.current_collection,
            help="Enter a new collection name or use existing one"
        )
        
        if st.session_state.available_collections:
            st.caption(f"Existing: {', '.join(st.session_state.available_collections)}")
        
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
        
        if st.button("Start Ingestion", use_container_width=True, type="primary", 
                     disabled=uploaded_file is None or not collection_name_input):
            if uploaded_file and collection_name_input:
                collection_name_input = collection_name_input.strip()
                
                if len(collection_name_input) < 3:
                    st.error("Collection name must be at least 3 characters long")
                elif not collection_name_input.replace("_", "").replace("-", "").replace(".", "").isalnum():
                    st.error("Collection name can only contain letters, numbers, dots, underscores, and hyphens")
                else:
                    try:
                        with st.spinner("Starting ingestion..."):
                            result = ingest_pdf_api(uploaded_file, collection_name_input, chunking_strategy)
                        
                        from datetime import datetime
                        ingestion_id = result["ingestion_id"]
                        st.session_state.active_ingestions.append({
                            "id": ingestion_id,
                            "filename": uploaded_file.name,
                            "collection": collection_name_input,
                            "strategy": chunking_strategy,
                            "started_at": datetime.now().strftime("%H:%M:%S")
                        })
                        
                        st.success("🎉 Ingestion Started!")
                        st.info(f"📌 Ingestion ID: `{ingestion_id}`")
                        st.info("💡 Processing in background. Check status below!")
                        
                    except Exception as e:
                        st.error(f"❌ Error starting ingestion: {str(e)}")
        
        st.divider()
        
        st.subheader("📊 Active Ingestions")
        
        if not st.session_state.active_ingestions:
            st.info("No active ingestions. Upload a document above to get started!")
        else:
            for idx, ing in enumerate(st.session_state.active_ingestions):
                with st.expander(
                    f"📄 {ing['filename']} ({ing['collection']}) - Started: {ing['started_at']}", 
                    expanded=True
                ):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        status_data = check_ingestion_status_api(ing["id"])
                        
                        status = status_data.get("status", "unknown")
                        message = status_data.get("message", "No message")
                        progress = status_data.get("progress", 0)
                        
                        if status == "pending":
                            st.info(f"⏳ {message}")
                        elif status == "processing":
                            st.warning(f"🔄 {message}")
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
                        
                        st.caption(f"ID: `{ing['id']}`")
                        st.caption(f"Strategy: {ing['strategy']}")
                    
                    with col2:
                        if st.button("🔄", key=f"refresh_{idx}", help="Refresh status"):
                            st.rerun()
                        
                        st.write("")
                        
                        if status in ["completed", "failed", "error"]:
                            if st.button("✕", key=f"remove_{idx}", help="Remove from list"):
                                st.session_state.active_ingestions.pop(idx)
                                st.rerun()
            
            st.divider()
            
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
            
            if auto_refresh:
                import time
                time.sleep(5)
                st.rerun()
    
    st.divider()

# ========== MAIN CONTENT AREA ==========

# Check API health first
if not check_api_health():
    st.error("⚠️ FastAPI backend is not running!")
    st.info("Start the API with: `python -m uvicorn backend.api:app --reload`")
    st.stop()

# Get API health status
try:
    health_response = requests.get(f"{API_URL}/health").json()
    
    if not health_response["database_exists"]:
        st.warning("⚠️ No vector database found. Please upload a document in the **Ingestion** tab first!")
    
    # ========== EVALUATION TAB (MAIN CONTENT) ==========
    if st.session_state.sidebar_tab == "Evaluation":
        st.subheader("📊 System Evaluation")
        
        st.info("🎯 Evaluate your RAG system's performance using test datasets from Langfuse")
        
        eval_type = st.radio(
            "Select Evaluation Type",
            ["Retrieval Quality", "Answer Quality"],
            help="Retrieval: Tests chunk retrieval accuracy | Answer: Tests complete answer quality using LLM-as-judge"
        )
        
        st.divider()
        
        # === RETRIEVAL QUALITY EVALUATION ===
        if eval_type == "Retrieval Quality":
            st.markdown("### 🔍 Retrieval Quality Evaluation")
            st.caption("Tests how well the system retrieves relevant document chunks across different k-values (3, 5, 7, 10)")
            
            dataset_name = st.text_input(
                "Langfuse Dataset Name",
                placeholder="e.g., research_paper_qa",
                help="Name of the dataset stored in Langfuse with test questions and expected answers",
                key="retrieval_dataset"
            )
            
            col1, col2 = st.columns([3, 1])
            with col1:
                run_retrieval = st.button("🚀 Run Retrieval Evaluation", type="primary", disabled=not dataset_name)
            with col2:
                if st.session_state.eval_results:
                    if st.button("🗑️ Clear Results"):
                        st.session_state.eval_results = None
                        st.rerun()
            
            if run_retrieval and dataset_name:
                st.session_state.eval_running = True
                
                with st.spinner(f"🔄 Evaluating retrieval quality for '{dataset_name}'... This may take 1-2 minutes."):
                    try:
                        results = evaluate_retrieval_api(
                            dataset_name=dataset_name,
                            collection_name=st.session_state.current_collection
                        )
                        
                        st.session_state.eval_results = results
                        st.session_state.eval_running = False
                        st.success("✅ Evaluation Complete!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.session_state.eval_running = False
            
            # Display results
            if st.session_state.eval_results and st.session_state.eval_results.get("success"):
                results = st.session_state.eval_results["results"]
                recommended_k = st.session_state.eval_results["recommended_k"]
                
                st.divider()
                st.markdown("### 📈 Results")
                
                st.success(f"🎯 **Recommended k-value:** {recommended_k}")
                st.caption("This k-value retrieved the most relevant chunks across all test questions")
                
                cols = st.columns(len(results))
                for idx, (k, data) in enumerate(results.items()):
                    with cols[idx]:
                        is_best = (int(k) == recommended_k)
                        delta = "Best!" if is_best else None
                        st.metric(
                            label=f"k={k}",
                            value=f"{data['average']:.3f}",
                            delta=delta,
                            help=f"Average relevance score across {data['count']} questions"
                        )
                
                st.markdown("#### Score Comparison")
                chart_data = {f"k={k}": data['average'] for k, data in results.items()}
                st.bar_chart(chart_data)
                
                with st.expander("📊 Detailed Breakdown by K-Value"):
                    for k, data in results.items():
                        st.markdown(f"**k={k}** {'⭐ Recommended' if int(k) == recommended_k else ''}")
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.write(f"- Questions: {data['count']}")
                            st.write(f"- Avg Score: {data['average']:.3f}")
                        with col2:
                            st.progress(data['average'], text=f"{data['average']*100:.1f}%")
                        st.write("")
        
        # === ANSWER QUALITY EVALUATION ===
        else:
            st.markdown("### 💬 Answer Quality Evaluation")
            st.caption("Uses LLM-as-judge (Cerebras llama3.3-70b) to evaluate complete answer quality")
            
            col1, col2 = st.columns(2)
            
            with col1:
                dataset_name = st.text_input(
                    "Langfuse Dataset Name",
                    placeholder="e.g., research_paper_qa",
                    help="Name of the dataset stored in Langfuse",
                    key="answer_dataset"
                )
            
            with col2:
                k_value = st.number_input(
                    "k-value",
                    min_value=1,
                    max_value=10,
                    value=5,
                    help="Number of chunks to retrieve for each question"
                )
            
            st.warning("⏱️ This evaluation may take 5-10 minutes as it generates and evaluates complete answers using LLM-as-a-judge")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                run_answer = st.button("🚀 Run Answer Evaluation", type="primary", disabled=not dataset_name)
            with col2:
                if st.session_state.eval_results:
                    if st.button("🗑️ Clear", key="clear_answer"):
                        st.session_state.eval_results = None
                        st.rerun()
            
            if run_answer and dataset_name:
                st.session_state.eval_running = True
                
                # Simple progress indicator (no threading - Streamlit doesn't support it)
                with st.spinner(f"🔄 Evaluating {dataset_name} with k={k_value}... This may take 5-10 minutes."):
                    try:
                        results = evaluate_answers_api(
                            dataset_name=dataset_name,
                            collection_name=st.session_state.current_collection,
                            k=k_value
                        )
                        
                        st.session_state.eval_results = results
                        st.session_state.eval_running = False
                        # Don't rerun - let Streamlit naturally refresh to show results
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.session_state.eval_running = False
            
            # Display results
            if st.session_state.eval_results and st.session_state.eval_results.get("success"):
                st.success("✅ Evaluation Complete!")
                
                averages = st.session_state.eval_results["averages"]
                scores = st.session_state.eval_results["scores"]
                
                st.divider()
                st.markdown("### 📈 Average Scores")
                st.caption(f"Evaluated across {len(scores)} questions")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Correctness", f"{averages['correctness']:.3f}",
                             help="Are the facts accurate?")
                with col2:
                    st.metric("Completeness", f"{averages['completeness']:.3f}",
                             help="Does it answer all parts?")
                with col3:
                    st.metric("Relevance", f"{averages['relevance']:.3f}",
                             help="Is it focused on the question?")
                with col4:
                    st.metric("Overall Quality", f"{averages['overall']:.3f}",
                             delta="Combined Score", delta_color="off")
                
                st.markdown("#### Score Breakdown")
                chart_data = {
                    "Correctness": averages['correctness'],
                    "Completeness": averages['completeness'],
                    "Relevance": averages['relevance']
                }
                st.bar_chart(chart_data)
                
                with st.expander(f"📊 Detailed Scores per Question ({len(scores)} questions)"):
                    for i, score in enumerate(scores, 1):
                        col1, col2, col3, col4 = st.columns(4)
                        col1.write(f"**Q{i}**")
                        col2.write(f"✓ {score['correctness']:.2f}")
                        col3.write(f"📝 {score['completeness']:.2f}")
                        col4.write(f"🎯 {score['relevance']:.2f}")
                        
                        overall = score['overall']
                        if overall >= 0.8:
                            st.success(f"Overall: {overall:.2f} - Excellent")
                        elif overall >= 0.6:
                            st.info(f"Overall: {overall:.2f} - Good")
                        else:
                            st.warning(f"Overall: {overall:.2f} - Needs Improvement")
                        st.write("")
                
                st.info("💡 All detailed traces and scores have been logged to Langfuse for deeper analysis")
    
    # ========== CHAT INTERFACE (MAIN CONTENT) ==========
    elif st.session_state.sidebar_tab == "Chat History":
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
            
            # Show chunks and evaluation for assistant messages
            if message["role"] == "assistant":
                # Find the corresponding user question
                user_question = ""
                if idx > 0 and current_messages[idx-1]["role"] == "user":
                    user_question = current_messages[idx-1]["content"]
                
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    # Existing chunk display
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
                
                with col2:
                    # NEW: Evaluate button
                    eval_key = f"eval_{st.session_state.current_session}_{idx}"
                    if st.button("📊 Evaluate", key=f"btn_{eval_key}", help="Evaluate this answer quality"):
                        with st.spinner("Evaluating answer..."):
                            try:
                                eval_result = evaluate_single_answer_api(
                                    question=user_question,
                                    answer=message["content"]
                                )
                                st.session_state.answer_scores[eval_key] = eval_result
                                st.rerun()
                            except Exception as e:
                                st.error(f"Evaluation failed: {str(e)}")
                
                # Display evaluation scores if they exist
                if eval_key in st.session_state.answer_scores:
                    scores = st.session_state.answer_scores[eval_key]
                    with st.expander("📊 Answer Quality Scores", expanded=True):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Correctness", f"{scores['correctness']:.2f}")
                        with col2:
                            st.metric("Completeness", f"{scores['completeness']:.2f}")
                        with col3:
                            st.metric("Relevance", f"{scores['relevance']:.2f}")
                        with col4:
                            overall = scores['overall']
                            st.metric("Overall", f"{overall:.2f}")
                        
                        if overall >= 0.8:
                            st.success("✅ Excellent answer quality!")
                        elif overall >= 0.6:
                            st.info("✓ Good answer quality")
                        else:
                            st.warning("⚠️ Answer could be improved")
                        
                        st.caption("Evaluated using LLM-as-judge (Cerebras llama3.3-70b)")
                
                assistant_response_count += 1
        
        # Process pending query if exists
        if st.session_state.pending_query and st.session_state.pending_session == st.session_state.current_session:
            query = st.session_state.pending_query
            
            with st.spinner(f"Thinking... (Using collection: {st.session_state.current_collection}, retrieving {st.session_state.retrieval_k} chunks)"):
                try:
                    result = query_api(query, st.session_state.current_collection, st.session_state.retrieval_k)
                    response = result["answer"]
                    source_chunks = result["chunks"]
                    st.session_state.last_error = None
                    st.session_state.last_query_k = st.session_state.retrieval_k
                    
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
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(response)
            
            # Store chunks and response
            if st.session_state.current_session not in st.session_state.chat_chunks:
                st.session_state.chat_chunks[st.session_state.current_session] = []
            st.session_state.chat_chunks[st.session_state.current_session].append(source_chunks)
            
            current_messages.append({"role": "assistant", "content": response})
            
            st.session_state.pending_query = None
            st.session_state.pending_session = None
            st.rerun()
        
        # Chat input
        if prompt := st.chat_input("Ask a question..."):
            current_messages.append({"role": "user", "content": prompt})
            st.session_state.pending_query = prompt
            st.session_state.pending_session = st.session_state.current_session
            st.rerun()

except Exception as e:
    st.error(f"Error communicating with API: {str(e)}")
