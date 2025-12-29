import chromadb

def fix_research_meta():
    print("Updating 'research_paper' metadata...")
    client = chromadb.PersistentClient(path="./Vector_DB")
    
    try:
        col = client.get_collection("research_paper")
        # Explicit negation to avoid confusion
        new_desc = "Academic research paper about InJET and engineering topics. Contains NO information about TechAxis Nepal training institute."
        
        col.modify(metadata={"description": new_desc})
        print(f"✓ Updated 'research_paper' to: {new_desc}")
        
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    fix_research_meta()
