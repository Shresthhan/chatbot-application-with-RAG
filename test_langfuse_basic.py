"""
test_langfuse_basic.py - Most basic Langfuse 3.x test
"""

print("Testing Langfuse 3.x...")

# Test 1: Import
print("\n1. Import CallbackHandler")
from langfuse.langchain import CallbackHandler
print("   ✓ Import successful")

# Test 2: Create handler
print("\n2. Create handler")
handler = CallbackHandler()
print("   ✓ Handler created")

# Test 3: Simple LangChain test
print("\n3. Test with LangChain")
from langchain_core.messages import HumanMessage
from langchain_cerebras import ChatCerebras

llm = ChatCerebras(
    model="llama-3.3-70b",
    temperature=0,
    api_key="csk-nmj4dx46c3hmk2ntmyttwcd9yecmk3d3rnhpvfrdhpfe3r43"
)
messages = [HumanMessage(content="Say 'success'")]

response = llm.invoke(messages, config={"callbacks": [handler]})
print(f"   ✓ LLM response: {response.content}")

# Flush
handler.langfuse.flush()
print("   ✓ Flushed to Langfuse")

print("\n✓ ALL TESTS PASSED!")
print("Langfuse 3.x is working correctly")
