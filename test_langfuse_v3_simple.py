"""
test_langfuse_v3_simple.py - Simple test for Langfuse 3.x LangChain integration
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=" * 80)
print("LANGFUSE 3.X SIMPLE TEST")
print("=" * 80)

# Test 1: Import
print("\nTEST 1: Import Langfuse 3.x LangChain CallbackHandler")
try:
    from langfuse.langchain import CallbackHandler
    print("✓ Import successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    exit(1)

# Test 2: Create handler
print("\nTEST 2: Create CallbackHandler")
print(f"  LANGFUSE_HOST: {os.getenv('LANGFUSE_HOST')}")
print(f"  LANGFUSE_PUBLIC_KEY: {os.getenv('LANGFUSE_PUBLIC_KEY', '')[:20]}...")
print(f"  LANGFUSE_SECRET_KEY: {'*' * 20}...")

try:
    handler = CallbackHandler()
    print("✓ CallbackHandler created")
except Exception as e:
    print(f"✗ Failed: {e}")
    exit(1)

# Test 3: Use with LangChain
print("\nTEST 3: Use CallbackHandler with LangChain LLM")
try:
    from langchain_core.messages import HumanMessage
    from backend.query import get_llm
    
    llm = get_llm()
    messages = [HumanMessage(content="Say 'Test passed!'")]
    
    # Invoke with callback
    response = llm.invoke(messages, config={"callbacks": [handler]})
    print(f"✓ LLM invoked with Langfuse callback")
    print(f"  Response: {response.content}")
    
    # Flush to ensure data is sent
    handler.langfuse.flush()
    print("✓ Data flushed to Langfuse")
    
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Summary
print("\n" + "=" * 80)
print("✓ ALL TESTS PASSED")
print("\nLangfuse 3.x integration is working!")
print("\nNext steps:")
print("1. Start backend: uvicorn backend.api:app --reload")
print("2. Make agent queries via API")
print("3. View traces in Langfuse UI: http://localhost:3000")
print("=" * 80)
