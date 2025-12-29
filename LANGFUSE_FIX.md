# Langfuse Installation Fix

## Problem
Langfuse version 3.9+ has breaking API changes that cause import errors:
```
ImportError: cannot import name 'TraceContext' from 'langfuse.types'
ModuleNotFoundError: No module named 'langfuse.callback'
```

## Temporary Solution (Current Status)
✅ **Langfuse is temporarily disabled** to allow the application to run.  
The agent will work normally but without observability tracing.

## Permanent Fix - Install Compatible Langfuse Version

### Option 1: Use Langfuse 2.x (Recommended)
The code was written for Langfuse 2.x API which is stable:

```bash
# Activate virtual environment
.venv\Scripts\activate

# Uninstall current version
pip uninstall -y langfuse

# Install compatible version
pip install "langfuse>=2.40.0,<3.0"

# Verify installation
python -c "from langfuse import Langfuse; from langfuse.callback import CallbackHandler; print('✓ Langfuse working!')"
```

### Option 2: Update Code for Langfuse 3.x API
If you want to use the latest Langfuse, the code needs to be updated because the API changed significantly:

**Changes in Langfuse 3.x:**
- `langfuse.callback.CallbackHandler` was moved/renamed
- Trace API changed from `client.trace()` to different pattern
- Types module structure changed

This would require rewriting the tracing code in:
- `backend/agent.py` (lines 14-70+)
- `backend/api.py` (lines 18-25)  
- `backend/query.py` (lines 17-32)

## Re-enabling Langfuse After Fix

Once you've installed a compatible version, revert these files:

### 1. backend/agent.py (lines 14-18)
Change from:
```python
# Langfuse temporarily disabled - package version incompatibility
# Reinstall: pip install "langfuse>=2.40.0,<3.0"
LANGFUSE_AVAILABLE = False
langfuse_client = None
print("[AGENT] ⚠ Langfuse temporarily disabled - agent will work without tracing")
```

Back to:
```python
try:
    from langfuse import Langfuse
    from langfuse.callback import CallbackHandler
    LANGFUSE_AVAILABLE = True
    langfuse_client = Langfuse()
    print("[AGENT] ✓ Langfuse initialized for agent tracing")
except ImportError:
    LANGFUSE_AVAILABLE = False
    langfuse_client = None
    print("[AGENT] ⚠ Langfuse not available, running without tracing")
except Exception as e:
    LANGFUSE_AVAILABLE = False
    langfuse_client = None
    print(f"[AGENT] ⚠ Langfuse initialization failed: {e}")
```

### 2. backend/api.py (lines 18-24)
Change from:
```python
# Temporarily disabled - package version incompatibility
CallbackHandler = None
Langfuse = None
LANGFUSE_AVAILABLE = False
```

Back to:
```python
try:
    from langfuse.callback import CallbackHandler
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError as e:
    print(f"⚠ Langfuse import failed: {e}. Continuing without Langfuse.")
    CallbackHandler = None
    Langfuse = None
    LANGFUSE_AVAILABLE = False
```

### 3. backend/query.py (lines 17-19)
Change from:
```python
# Langfuse temporarily disabled - package version incompatibility
CallbackHandler = None
LANGFUSE_AVAILABLE = False
```

Back to:
```python
try:
    from langfuse.callback import CallbackHandler
    LANGFUSE_AVAILABLE = True
except ImportError as e:
    print(f"⚠ Langfuse import failed in query.py: {e}")
    CallbackHandler = None
    LANGFUSE_AVAILABLE = False
```

### 4. experiments/evaluate_rag.py (lines 11-13)
Change from:
```python
# Langfuse temporarily disabled - package version incompatibility
Langfuse = None
LANGFUSE_AVAILABLE = False
```

Back to:
```python
try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError as e:
    print(f"⚠ Langfuse import failed in evaluate_rag.py: {e}")
    Langfuse = None
    LANGFUSE_AVAILABLE = False
```

### 5. experiments/evaluate_answers.py (lines 12-14)
Change from:
```python
# Langfuse temporarily disabled - package version incompatibility
Langfuse = None
LANGFUSE_AVAILABLE = False
```

Back to:
```python
try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError as e:
    print(f"⚠ Langfuse import failed in evaluate_answers.py: {e}")
    Langfuse = None
    LANGFUSE_AVAILABLE = False
```

## Testing After Re-enabling

```bash
# Start backend
python.exe -m uvicorn backend.api:app --reload --port 8000

# Should see:
# [AGENT] ✓ Langfuse initialized for agent tracing
# INFO:     Uvicorn running on http://127.0.0.1:8000

# Test a query
curl -X POST http://localhost:8000/react_agent_query \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}'

# Check Langfuse dashboard for traces
```

## Environment Variables Required

Make sure these are set in your `.env` file:
```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

## Summary

| Status | Action |
|--------|--------|
| ✅ Current | Agent works without Langfuse tracing |
| 🔧 To Fix | Install `pip install "langfuse>=2.40.0,<3.0"` |
| 📝 Then | Revert code changes in 5 files above |
| ✅ Result | Full observability tracing enabled |

See [LANGFUSE_TRACING_GUIDE.md](LANGFUSE_TRACING_GUIDE.md) for complete tracing implementation details.
