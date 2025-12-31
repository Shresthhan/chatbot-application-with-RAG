# Langfuse 3.x Integration - Setup Complete

## Summary

Successfully integrated **Langfuse 3.x** for agent observability and tracing. The integration uses Langfuse's modern LangChain callback handler.

## What Changed

### 1. Package Version
- **Upgraded**: `langfuse-2.60.10` → `langfuse-3.11.2`
- **Reason**: Langfuse 2.x requires old LangChain structure (`langchain.callbacks.base`), but we have modern modular LangChain (`langchain-core`, `langchain-community`, etc.)

### 2. Integration Approach
Instead of using the complex trace API, we use the simple **CallbackHandler** pattern:

```python
from langfuse.langchain import CallbackHandler

# Create handler (automatically uses env variables)
handler = CallbackHandler()

# Pass to LLM/chain
response = llm.invoke(messages, config={"callbacks": [handler]})

# Flush to ensure data sent
handler.langfuse.flush()
```

### 3. Files Modified

#### backend/api.py
```python
# Import
from langfuse.langchain import CallbackHandler

# In endpoint
langfuse_callback = CallbackHandler()
result = agent.invoke(request.question, langfuse_handler=langfuse_callback)
```

#### backend/langgraph_agent.py
```python
from langfuse.langchain import CallbackHandler

def invoke(self, query: str, langfuse_handler=None):
    # Pass callback to LLM config
    config = {
        "recursion_limit": 50,
        "callbacks": [langfuse_handler] if langfuse_handler else []
    }
```

#### backend/query.py
```python
from langfuse.langchain import CallbackHandler
```

#### experiments/evaluate_rag.py & evaluate_answers.py
- Already using `from langfuse import Langfuse` (correct for 3.x)

## Environment Variables Required

```env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3000
```

## How It Works

1. **CallbackHandler Creation**: When `/langgraph_agent_query` endpoint is called, a new `CallbackHandler()` is created
2. **Automatic Tracking**: The handler intercepts all LangChain/LangGraph events:
   - LLM calls (prompts, responses, tokens)
   - Tool invocations (inputs, outputs, timing)
   - Agent reasoning steps
   - Errors and exceptions
3. **Trace Organization**: Everything is grouped under a single trace session
4. **Flush**: Data is sent to Langfuse server (localhost:3000 or cloud)

## Testing

✅ **Basic Test Passed**:
```bash
.venv\Scripts\python test_langfuse_basic.py
```
- ✓ Import successful
- ✓ Handler created
- ✓ Integration with LangChain works

## Next Steps to Verify

1. **Start Langfuse Server** (if using local):
   ```bash
   docker-compose up
   ```
   Access UI: http://localhost:3000

2. **Start Backend**:
   ```bash
   uvicorn backend.api:app --reload
   ```

3. **Make Agent Query**:
   - Via Streamlit UI: `streamlit run frontend/app_api.py`
   - Via API directly: POST to `/langgraph_agent_query`

4. **Check Console**: You should see:
   ```
   [API] Langfuse callback created (trace session: <uuid>)
   ```

5. **View in Langfuse UI**: 
   - Navigate to http://localhost:3000
   - Check "Traces" section
   - You'll see detailed execution logs:
     - Agent reasoning steps
     - Tool calls (web_search, rag_query)
     - LLM interactions
     - Token usage
     - Latency metrics

## API Differences: Langfuse 2.x vs 3.x

### Langfuse 2.x (Old - Incompatible)
```python
from langfuse.callback import CallbackHandler  # ❌ Requires old langchain
from langfuse import Langfuse

client = Langfuse()
trace = client.trace(name="...", input={...})
handler = trace.get_langchain_handler()
```

### Langfuse 3.x (New - Current)
```python
from langfuse.langchain import CallbackHandler  # ✅ Works with modern langchain

handler = CallbackHandler()  # Simple!
# Uses env variables automatically
# No need for explicit trace creation
```

## Troubleshooting

### "Authentication error: Langfuse client initialized without public_key"
- **Solution**: Check `.env` file has correct `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`
- **Note**: This warning appears but handler still works if env vars are set properly

### Traces not appearing in UI
1. Ensure Langfuse server is running: `curl http://localhost:3000`
2. Check `LANGFUSE_HOST` matches your server URL
3. Verify handler.langfuse.flush() is called (happens automatically on app shutdown)

### "Connection refused" errors
- Langfuse server not running
- Wrong port (default is 3000)
- Firewall blocking connection

## Benefits

✅ **Full Observability**: See every step of agent execution  
✅ **Performance Metrics**: Token usage, latency, cost tracking  
✅ **Debugging**: Identify why agent makes certain decisions  
✅ **Evaluation**: Compare different runs, A/B testing  
✅ **Monitoring**: Production tracing and error tracking  

## Documentation

- **Langfuse Docs**: https://langfuse.com/docs
- **LangChain Integration**: https://langfuse.com/docs/integrations/langchain
- **Tracing Guide**: https://langfuse.com/docs/tracing

---

**Status**: ✅ Langfuse 3.x integration complete and ready to use!
