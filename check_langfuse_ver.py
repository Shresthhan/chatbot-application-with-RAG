try:
    from langfuse import Langfuse
    # Mocking client to see if method exists on the class/type hints or just trying to instantiate
    # But without credentials it might fail.
    # Let's inspect the library version and dir
    import langfuse
    print(f"Langfuse version: {langfuse.__version__}")
    
    # Check if we can instantiate a dummy trace
    # (Might fail if no network/keys, but we'll see)
    
except ImportError:
    print("Langfuse not installed")
except Exception as e:
    print(f"Error: {e}")
