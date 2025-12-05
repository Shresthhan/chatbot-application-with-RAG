# Data Folder

## Purpose

This folder is **optional** and only used for **manual CLI-based ingestion**.

## Usage Scenarios

### 1. UI-Based Ingestion (Recommended) ✅
- Use the Streamlit app's **Ingestion tab**
- Upload PDFs directly through the drag-and-drop interface
- No need for this folder at all
- Documents are appended to the vector database automatically

### 2. CLI-Based Ingestion (Manual)
- Place your PDF in this folder
- Update `PDF_PATH` in `ingest.py` configuration
- Run: `python ingest.py`
- Useful for batch processing or automation

## Current Status

- Contains: `Conference_paper_pdf .pdf` (example document)
- Can be deleted if you only use UI-based ingestion
- Can keep for CLI-based ingestion convenience

## Recommendation

**For most users**: You don't need this folder anymore since the Streamlit UI handles document uploads. Feel free to delete it or keep it for backup/manual processing purposes.
