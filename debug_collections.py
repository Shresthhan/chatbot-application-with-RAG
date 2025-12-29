import chromadb
import os
import json

def check_collections():
    CHROMA_PATH = "./Vector_DB"
    if not os.path.exists(CHROMA_PATH):
        print(f"Directory {CHROMA_PATH} does not exist.")
        return

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collections = client.list_collections()
    
    print(f"Total collections found: {len(collections)}")
    for col in collections:
        print(f"--- Collection: {col.name} ---")
        print(f"  Count: {col.count()}")
        meta = col.metadata or {}
        print(f"  DESCRIPTION: {meta.get('description', 'NO DESCRIPTION')}")
        print(f"  Metadata: {json.dumps(meta, indent=2)}")
        if col.count() > 0:
            try:
                # peek at first chunk
                peek = col.peek(1)
                if peek and peek.get('documents'):
                    print(f"  Snippet: {peek['documents'][0][:200]}...")
            except Exception as e:
                print(f"  Could not peek: {e}")

if __name__ == "__main__":
    check_collections()
