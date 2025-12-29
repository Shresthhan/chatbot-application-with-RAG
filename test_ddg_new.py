from duckduckgo_search import DDGS
import json

def test_enhanced_search(query):
    print(f"Testing enhanced search for: {query}")
    try:
        results = []
        with DDGS() as ddgs:
            ddg_gen = ddgs.text(query, max_results=5)
            for r in ddg_gen:
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")
                })
        
        print(f"Found {len(results)} results")
        for i, r in enumerate(results, 1):
            print(f"{i}. {r['title']} - {r['url']}")
            print(f"   Snippet: {r['snippet'][:100]}...")
            
    except Exception as e:
        print(f"Error: {e}")

test_enhanced_search("What is the latest score for the Lakers game?")
