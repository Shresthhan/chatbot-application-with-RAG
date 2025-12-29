"""
tools.py - External tools for agent (web search, etc.)
"""

import os
import requests
from typing import List, Dict
from dotenv import load_dotenv

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

# Placeholder for future tools
def translate_query(query: str, target_lang: str = "en") -> str:
    """
    Translate query to English for better retrieval.
    TODO: Implement with Google Translate API or similar
    """
    # For now, return as-is
    return query
