"""
Test Langfuse connection and basic functionality
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_langfuse_import():
    """Test if Langfuse can be imported"""
    print("=" * 80)
    print("TEST 1: Import Langfuse")
    print("=" * 80)
    
    try:
        from langfuse import Langfuse
        from langfuse.callback import CallbackHandler
        print("✓ Langfuse imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import Langfuse: {e}")
        return False

def test_langfuse_connection():
    """Test connection to Langfuse server"""
    print("\n" + "=" * 80)
    print("TEST 2: Connect to Langfuse Server")
    print("=" * 80)
    
    try:
        from langfuse import Langfuse
        
        # Get config from environment
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        host = os.getenv("LANGFUSE_HOST")
        
        print(f"Host: {host}")
        print(f"Public Key: {public_key[:20]}..." if public_key else "No public key")
        print(f"Secret Key: {'*' * 20}..." if secret_key else "No secret key")
        
        # Initialize Langfuse
        langfuse = Langfuse()
        print("✓ Langfuse client initialized")
        
        # Test trace creation
        trace = langfuse.trace(
            name="test_connection",
            input={"test": "hello"}
        )
        print(f"✓ Test trace created: {trace.id}")
        
        # Flush to ensure data is sent
        langfuse.flush()
        print("✓ Data flushed to server")
        
        return True
        
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_langfuse_with_langchain():
    """Test Langfuse with LangChain callback"""
    print("\n" + "=" * 80)
    print("TEST 3: Langfuse with LangChain Callback")
    print("=" * 80)
    
    try:
        from langfuse import Langfuse
        from langfuse.callback import CallbackHandler
        
        # Initialize
        langfuse = Langfuse()
        trace = langfuse.trace(
            name="test_langchain_callback",
            input={"query": "test query"}
        )
        
        # Get LangChain handler
        handler = trace.get_langchain_handler()
        print(f"✓ LangChain callback handler created")
        print(f"  Handler type: {type(handler)}")
        print(f"  Trace ID: {trace.id}")
        
        # Flush
        langfuse.flush()
        
        return True
        
    except Exception as e:
        print(f"✗ LangChain callback test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("LANGFUSE CONNECTION TEST")
    print("="*80 + "\n")
    
    results = []
    
    # Test 1: Import
    results.append(("Import", test_langfuse_import()))
    
    # Test 2: Connection
    results.append(("Connection", test_langfuse_connection()))
    
    # Test 3: LangChain integration
    results.append(("LangChain Callback", test_langfuse_with_langchain()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:.<40} {status}")
    
    all_passed = all(result for _, result in results)
    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL TESTS PASSED - Langfuse is ready to use!")
    else:
        print("✗ SOME TESTS FAILED - Check the errors above")
        print("\nTroubleshooting:")
        print("1. Make sure Langfuse server is running on http://localhost:3000")
        print("2. Check your .env file has correct LANGFUSE_* variables")
        print("3. Verify langfuse package is installed: pip show langfuse")
    print("="*80)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
