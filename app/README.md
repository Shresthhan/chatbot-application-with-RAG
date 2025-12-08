# Legacy App

This folder contains the original Streamlit application that runs without the FastAPI backend.

## Files

- **app.py** - Standalone Streamlit application
  - Direct imports from backend modules
  - No API communication
  - Single collection only
  - Synchronous ingestion

## Note

This is the legacy version kept for reference. The recommended approach is to use:
- **backend/api.py** + **frontend/app_api.py** for the full multi-collection experience

## Running

From project root:
```bash
streamlit run app\app.py
```

Or use: `start_app.bat`

## Limitations

- No background ingestion
- No job tracking
- Single collection only
- Blocks UI during ingestion
