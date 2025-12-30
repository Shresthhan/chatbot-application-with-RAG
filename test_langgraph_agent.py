"""
test_langgraph_agent.py - Test script for LangGraph multi-collection agent
Run this script to verify the agent is working correctly
"""

import requests
import json
import sys

BASE_URL = "http://localhost:8000"

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}\n")

def test_collections_list():
    """Test listing collections"""
    print_section("TEST 1: List Collections")
    
    response = requests.get(f"{BASE_URL}/collections/list")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Found {data['total']} collections:")
        for col in data['collections']:
            print(f"  - {col['name']}: {col['document_count']} documents")
            print(f"    Description: {col['description'][:100]}...")
        return True
    else:
        print(f"✗ Failed: {response.status_code}")
        print(response.text)
        return False

def test_create_collection():
    """Test creating a new collection"""
    print_section("TEST 2: Create Test Collection")
    
    data = {
        "collection_name": "test_collection",
        "description": """**Purpose:** Test collection for agent verification.

**Use this tool for:**
- Test queries and verification
- Example questions
- Agent testing

**Example queries:**
- "Test the agent"
- "Verify collection access"
"""
    }
    
    response = requests.post(f"{BASE_URL}/collections/create", data=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Collection created: {result['collection_name']}")
        return True
    else:
        print(f"⚠ Response: {response.status_code}")
        print(response.text)
        return False

def test_agent_query(question):
    """Test the LangGraph agent with a query"""
    print_section(f"TEST 3: Agent Query")
    print(f"Question: {question}\n")
    
    payload = {
        "question": question,
        "k": 5
    }
    
    response = requests.post(
        f"{BASE_URL}/langgraph_agent_query",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        
        print(f"✓ Answer received:")
        print(f"\n{result['answer']}\n")
        
        print(f"Tools used: {', '.join(result.get('tools_used', []))}")
        
        print(f"\nIntermediate steps:")
        for i, step in enumerate(result.get('intermediate_steps', []), 1):
            print(f"  Step {i}: {step['tool']}")
            print(f"    Input: {step['input']}")
            print(f"    Output: {step['output'][:100]}...")
        
        return True
    else:
        print(f"✗ Failed: {response.status_code}")
        print(response.text)
        return False

def test_web_search_query():
    """Test a query that should trigger web search"""
    print_section("TEST 4: Web Search Query")
    
    question = "What are the latest news in AI today?"
    print(f"Question: {question}\n")
    
    payload = {
        "question": question,
        "k": 5
    }
    
    response = requests.post(
        f"{BASE_URL}/langgraph_agent_query",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        
        tools_used = result.get('tools_used', [])
        
        if 'web_search' in tools_used:
            print("✓ Agent correctly used web search for current events")
            print(f"\nAnswer: {result['answer'][:200]}...")
            return True
        else:
            print(f"⚠ Agent used: {tools_used}")
            print("Expected web_search but got different tool")
            return False
    else:
        print(f"✗ Failed: {response.status_code}")
        print(response.text)
        return False

def main():
    """Run all tests"""
    print_section("LangGraph Agent Test Suite")
    print("Testing your multi-collection RAG agent...\n")
    print("Prerequisites:")
    print("  1. Backend should be running: python -m uvicorn backend.api:app --reload")
    print("  2. At least one collection should exist with documents")
    print("  3. TAVILY_API_KEY should be set in .env for web search")
    
    input("\nPress Enter to start tests...")
    
    # Run tests
    results = []
    
    # Test 1: List collections
    results.append(("List Collections", test_collections_list()))
    
    # Test 2: Create test collection (optional, may fail if exists)
    results.append(("Create Collection", test_create_collection()))
    
    # Test 3: Agent query (modify based on your collections)
    test_query = "What information do you have in your knowledge base?"
    results.append(("Agent Query", test_agent_query(test_query)))
    
    # Test 4: Web search query
    results.append(("Web Search", test_web_search_query()))
    
    # Summary
    print_section("TEST SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your LangGraph agent is working correctly.")
    else:
        print("\n⚠ Some tests failed. Check the output above for details.")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nTests cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
