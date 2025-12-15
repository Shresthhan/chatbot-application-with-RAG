import requests
import json

API_URL = "http://localhost:8000"

def verify():
    print("Verifying trace_id in query response...")
    try:
        # First check health
        health = requests.get(f"{API_URL}/health")
        if health.status_code != 200:
            print("❌ API is not healthy")
            return

        cols = requests.get(f"{API_URL}/collections").json()
        if not cols['collections']:
            print("⚠ No collections found to test with.")
            # Can't fully test query without collection
            return
        
        col_name = cols['collections'][0]['name']
        print(f"Using collection: {col_name}")

        # Send a query
        resp = requests.post(f"{API_URL}/query", json={
            "question": "Test query after fix",
            "collection_name": col_name,
            "k": 1
        })
        
        if resp.status_code != 200:
            print(f"❌ Query failed: {resp.text}")
            return

        data = resp.json()
        trace_id = data.get("trace_id")
        
        if trace_id:
            print(f"✅ SUCCESS: Received trace_id: {trace_id}")
        else:
            print("❌ FAILURE: trace_id missing")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    verify()
