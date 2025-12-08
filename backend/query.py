# query_simple.py - Simple RAG query (no agent yet)

import warnings
warnings.filterwarnings("ignore")

import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

# Configuration (same as ingestion)
CHROMA_PATH = "./Vector_DB"
COLLECTION_NAME = "my_docss"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 1. LOAD EXISTING VECTOR DB
def load_vectordb():
    """Load the Chroma vector database we created (default collection)"""
    return load_vectordb_with_collection(COLLECTION_NAME)

def load_vectordb_with_collection(collection_name: str):
    """Load a specific collection from the vector database"""
    # print(f"Loading collection '{collection_name}'...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",  # Better quality
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectordb = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name=collection_name
    )
    # print(f"✓ Loaded collection '{collection_name}'")
    return vectordb

# 2. SETUP LLM
def get_llm():
    """Initialize Google Gemini LLM"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )
    return llm

# 3. CREATE RAG CHAIN
def create_rag_chain(vectordb, llm):
    """Create a Retrieval QA chain using LCEL (LangChain Expression Language)"""
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    
    # Define the prompt template
    template = """You are a helpful research assistant. 

    IMPORTANT: If the user's message is a greeting (like "hi", "hello", "hey", "how are you"), 
    respond warmly and naturally WITHOUT referring to any context. Ask how you can help them.
    
    For actual questions:
    - Use the following context from the research paper to answer the question
    - If the answer is in the context, provide a detailed response
    - If not explicitly stated but related information exists, provide what you can infer
    - If the context is not relevant to the question, say so clearly

Context:
{context}

Question: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create the RAG chain using LCEL
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

# 4. ASK QUESTIONS
def ask_question(rag_chain, retriever, question):
    """Query the RAG system"""
    # print(f"\nQuestion: {question}")
    # print("Searching and generating answer...\n")
    
    # Get the answer from the chain
    answer = rag_chain.invoke(question)
    
    # Get source documents separately
    sources = retriever.invoke(question)
    
    print(f"Answer: {answer}\n")
    # print(f"Sources: {len(sources)} document chunks used")
    return answer

def main():
    print("=== RAG Query System ===\n")
    
    # Setup
    vectordb = load_vectordb()
    llm = get_llm()
    rag_chain, retriever = create_rag_chain(vectordb, llm)
    
    print("(Type 'exit' or 'quit' to stop)\n")
    
    # Interactive query loop
    while True:
        # Get user input
        question = input("Your question: ").strip()
        
        # Check if user wants to exit
        if question.lower() in ['exit', 'quit', 'q']:
            print("\nGoodbye!")
            break
        
        # Skip empty questions
        if not question:
            print("Please enter a question.\n")
            continue
        
        # Ask the question
        ask_question(rag_chain, retriever, question)
        print("-" * 50)

if __name__ == "__main__":
    main()
