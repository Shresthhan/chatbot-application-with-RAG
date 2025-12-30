"""
tools.py - Dynamic tool generation for RAG collections and web search
Automatically creates tools from Qdrant collections with 3-part descriptions
"""

import os
import requests
from typing import List, Dict, Callable
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

from tavily import TavilyClient

# Initialize Tavily client with API key from environment
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def web_search_tool(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    Enhanced web search using Tavily API.
    Provides high-quality search results optimized for LLMs and RAG systems.
    
    Args:
        query: Search query
        num_results: Number of results to return
    
    Returns:
        List of search results with title, url, snippet
    """
    try:
        print(f"[TOOL] Searching web with Tavily for: {query}")
        
        # Perform Tavily search
        response = tavily_client.search(
            query=query,
            max_results=num_results
        )
        
        # Extract results
        results = []
        for r in response.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", "")
            })
        
        if not results:
            print("[TOOL] No web results found.")
            return [{
                "title": "No Results",
                "url": "",
                "snippet": f"No web search results found for '{query}'."
            }]
            
        return results
        
    except Exception as e:
        print(f"Web search error: {e}")
        return [{
            "title": "Search Error",
            "url": "",
            "snippet": f"Web search failed: {str(e)}"
        }]


# ============================================================================
# DYNAMIC TOOL GENERATION FOR COLLECTIONS
# ============================================================================

def create_collection_search_tool(collection_name: str, description: str, k: int = 5):
    """
    Dynamically create a search tool for a specific Qdrant collection.
    Uses LangChain's @tool decorator for proper integration with agents.
    
    Args:
        collection_name: Name of the Qdrant collection
        description: Human-readable description (3-part format recommended)
        k: Number of chunks to retrieve
    
    Returns:
        LangChain Tool instance
    """
    from backend.query import load_qdrant_vectordb, get_llm
    
    # Create the tool description with collection name
    tool_description = f"Search the '{collection_name}' knowledge base.\n\n{description}"
    
    @tool(description=tool_description)
    def search_tool(query: str) -> str:
        """Search the collection for relevant information."""
        try:
            print(f"[TOOL] Searching collection '{collection_name}' for: {query}")
            
            # Load the collection's vector database
            vectordb = load_qdrant_vectordb(collection_name)
            retriever = vectordb.as_retriever(search_kwargs={"k": k})
            
            # Retrieve relevant documents (using invoke for newer LangChain versions)
            docs = retriever.invoke(query)
            
            if not docs:
                return f"No relevant information found in '{collection_name}' for the query: {query}"
            
            # Format results
            result = f"Found {len(docs)} relevant documents in '{collection_name}':\n\n"
            for i, doc in enumerate(docs, 1):
                content = doc.page_content[:500]  # Limit length
                metadata = doc.metadata
                result += f"Document {i}:\n{content}\n"
                if metadata:
                    result += f"Source: {metadata.get('source', 'Unknown')}\n"
                result += "\n"
            
            return result
            
        except Exception as e:
            error_msg = f"Error searching collection '{collection_name}': {str(e)}"
            print(f"[ERROR] {error_msg}")
            return error_msg
    
    # Set the tool name to be unique for each collection
    search_tool.name = f"search_{collection_name}"
    
    return search_tool


def generate_tool_description(collection_name: str, base_description: str) -> str:
    """
    Generate a structured 3-part tool description from basic description.
    
    Format:
    1. Purpose: What the tool does
    2. Use for: Keywords and topics (helps agent decide when to use)
    3. Examples: Sample query patterns
    
    Args:
        collection_name: Name of collection
        base_description: User-provided basic description
    
    Returns:
        Enhanced 3-part description
    """
    # Extract key topics from description
    description_lower = base_description.lower()
    
    # Build structured description
    structured = f"""**Purpose:** {base_description}

**Use this tool for:**
- Questions about topics in the '{collection_name}' collection
- Queries requiring information from uploaded documents
- Domain-specific knowledge searches

**Example queries:**
- "What does {collection_name.replace('_', ' ')} say about..."
- "Find information on... in {collection_name.replace('_', ' ')}"
- "According to {collection_name.replace('_', ' ')}..."
"""
    
    return structured


def create_web_search_tool():
    """
    Create the web search tool with structured description.
    Returns a LangChain Tool instance.
    """
    @tool
    def web_search(query: str) -> str:
        """Search the internet for current information and recent events.
        
        **Purpose:** Search the web for up-to-date information not available in local collections.
        
        **Use this tool for:**
        - Current events and recent news
        - Real-time information and latest updates
        - Information not found in knowledge base collections
        - Fact-checking and verification
        - Broad general knowledge queries
        
        **Example queries:**
        - "What are the latest developments in..."
        - "Current news about..."
        - "What happened recently with..."
        - "Latest information on..."
        
        Args:
            query: The search query
        
        Returns:
            Web search results with sources
        """
        try:
            print(f"[TOOL] Searching web for: {query}")
            
            # Perform Tavily search
            response = tavily_client.search(
                query=query,
                max_results=5
            )
            
            # Extract and format results
            results = response.get("results", [])
            
            if not results:
                return f"No web search results found for: {query}"
            
            formatted_result = f"Found {len(results)} web results:\n\n"
            for i, r in enumerate(results, 1):
                title = r.get("title", "No title")
                url = r.get("url", "")
                content = r.get("content", "")[:400]  # Limit length
                
                formatted_result += f"Result {i}: {title}\n"
                formatted_result += f"URL: {url}\n"
                formatted_result += f"Content: {content}\n\n"
            
            return formatted_result
            
        except Exception as e:
            error_msg = f"Web search error: {str(e)}"
            print(f"[ERROR] {error_msg}")
            return error_msg
    
    return web_search


def get_all_tools() -> List:
    """
    Dynamically generate all available tools based on existing Qdrant collections.
    This function discovers collections and creates a tool for each one, plus web search.
    
    Returns:
        List of LangChain Tool instances
    """
    from backend.collection_manager import get_collection_manager
    
    tools = []
    
    # Get collection manager and discover collections
    manager = get_collection_manager()
    collections_metadata = manager.get_all_collections_metadata()
    
    print(f"\n[TOOLS] Discovered {len(collections_metadata)} collections")
    
    # Create a tool for each collection
    for collection_name, metadata in collections_metadata.items():
        description = metadata.get("description", f"Collection: {collection_name}")
        
        # Generate structured description if needed
        if not description.startswith("**Purpose:**"):
            description = generate_tool_description(collection_name, description)
        
        tool = create_collection_search_tool(collection_name, description)
        tools.append(tool)
        print(f"  ✓ Created tool for collection: {collection_name}")
    
    # Add web search tool
    web_tool = create_web_search_tool()
    tools.append(web_tool)
    print(f"  ✓ Created web search tool")
    
    print(f"[TOOLS] Total tools available: {len(tools)}\n")
    
    return tools


# Legacy function for backward compatibility
def web_search_tool(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    Legacy web search function (kept for backward compatibility).
    For new implementations, use create_web_search_tool() instead.
    """
    try:
        print(f"[TOOL] Searching web with Tavily for: {query}")
        
        response = tavily_client.search(
            query=query,
            max_results=num_results
        )
        
        results = []
        for r in response.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", "")
            })
        
        if not results:
            return [{
                "title": "No Results",
                "url": "",
                "snippet": f"No web search results found for '{query}'."
            }]
            
        return results
        
    except Exception as e:
        print(f"Web search error: {e}")
        return [{
            "title": "Search Error",
            "url": "",
            "snippet": f"Web search failed: {str(e)}"
        }]
