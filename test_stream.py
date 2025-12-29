import requests
import json
import sseclient

def test_stream():
    url = "http://127.0.0.1:8000/agent_query_stream"
    data = {
        "question": "what is techaxis about and what does it aim",
        "k": 3
    }
    
    print(f"Querying {url} with: '{data['question']}'...")
    
    try:
        response = requests.post(url, json=data, stream=True)
        client = sseclient.SSEClient(response)
        
        full_answer = ""
        for event in client.events():
            try:
                data = json.loads(event.data)
                ev_type = data.get("type")
                content = data.get("content")
                
                if ev_type == "status":
                    print(f"\n[STATUS] {content}", flush=True)
                elif ev_type == "routing":
                    msg = f"Tool: {content.get('tool')}, Collection: {content.get('collection')}"
                    print(f"\n[ROUTING DECISION] {msg}", flush=True)
                    with open("decision.txt", "w") as f:
                        f.write(msg)
                elif ev_type == "answer_chunk":
                    full_answer += content
                elif ev_type == "final_state":
                    print("\n[DONE]")
                elif ev_type == "error":
                    print(f"[ERROR] {content}")
            except Exception as e:
                # print(f"Error parsing event: {e}")
                pass
                
        # print(f"\n\nFULL ANSWER RECEIVED:\n{full_answer}")
        
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_stream()
