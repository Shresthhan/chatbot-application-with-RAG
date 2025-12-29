import chromadb
import json

def verify_view():
    print("--- Simulating Agent's View of Collections ---")
    client = chromadb.PersistentClient(path="./Vector_DB")
    all_collections = client.list_collections()
    
    for col in all_collections:
        if col.name == "research_paper":
            print(f"Collection: '{col.name}'")
            print(f"   Metadata Raw: {col.metadata}")

    print("\n--- Done ---")

if __name__ == "__main__":
    verify_view()
