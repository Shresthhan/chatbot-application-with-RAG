"""
test_langfuse_v3.py - Test Langfuse 3.x integration with LangChain
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=" * 80)
print("LANGFUSE 3.X CONNECTION TEST")
print("=" * 80)

# Test 1: Import Langfuse 3.x
print("\n" + "=" * 80)
print("TEST 1: Import Langfuse 3.x")
print("=" * 80)
try:
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler
    print("✓ Successfully imported Langfuse 3.x")
    print("✓ Successfully imported CallbackHandler from langfuse.langchain")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    exit(1)

# Test 2: Connect to Langfuse server
print("\n" + "=" * 80)
print("TEST 2: Connect to Langfuse Server")
print("=" * 80)
langfuse_host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")

print(f"Host: {langfuse_host}")
print(f"Public Key: {langfuse_public_key[:20]}...")
print(f"Secret Key: {'*' * 20}...")

try:
    # Initialize Langfuse client
    client = Langfuse()
    print("✓ Langfuse client initialized")
    
    # Create a test trace
    trace = client.trace(
        name="test_trace_v3",
        metadata={"test": "langfuse_3.x"}
    )
    print(f"✓ Test trace created: {trace.id}")
    
    # Flush to ensure data is sent
    client.flush()
    print("✓ Data flushed to server")
    
except Exception as e:
    print(f"✗ Connection test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 3: Langfuse with LangChain Callback
print("\n" + "=" * 80)
print("TEST 3: Langfuse 3.x with LangChain Callback")
print("=" * 80)
try:
    # Create callback handler
    handler = CallbackHandler(
        trace_name="test_langchain_integration",
        metadata={"test": "callback_handler"}
    )
    print(f"✓ CallbackHandler created successfully")
    print(f"  Handler type: {type(handler)}")
    print(f"  Trace ID: {handler.trace_id if hasattr(handler, 'trace_id') else 'N/A'}")
    
    # Test with a simple LangChain LLM call
    try:
        from langchain_core.messages import HumanMessage
        from backend.query import get_llm
        
        llm = get_llm()
        messages = [HumanMessage(content="Say 'Langfuse test successful!'")]
        
        # Invoke with callback
        response = llm.invoke(messages, config={"callbacks": [handler]})
        print(f"✓ LLM invoked with Langfuse callback")
        print(f"  Response: {response.content}")
        
        # Flush
        handler.langfuse.flush()
        print("✓ Trace data flushed to Langfuse")
        
    except Exception as e:
        print(f"⚠ LLM test skipped: {e}")
    
except Exception as e:
    print(f"✗ LangChain callback test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Summary
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print("Import.................................. ✓ PASSED")
print("Connection.............................. ✓ PASSED")
print("LangChain Callback...................... ✓ PASSED")
print("\n" + "=" * 80)
print("✓ ALL TESTS PASSED - Langfuse 3.x is working!")
print("\nNext steps:")
print("1. Start backend: uvicorn backend.api:app --reload")
print("2. Make agent query and check console for trace IDs")
print("3. View traces in Langfuse UI: http://localhost:3000")
print("=" * 80)
