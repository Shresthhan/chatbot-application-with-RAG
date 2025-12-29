import requests
import json

def test_search(query):
    url = "https://api.duckduckgo.com/"
    params = {
        "q": query,
        "format": "json",
        "no_html": 1,
        "skip_disambig": 1
    }
    response = requests.get(url, params=params, timeout=5)
    data = response.json()
    print(f"Results for: {query}")
    print(f"Abstract: {data.get('AbstractText')}")
    print(f"Related Topics Count: {len(data.get('RelatedTopics', []))}")
    if data.get('RelatedTopics'):
        print(f"First topic: {data['RelatedTopics'][0] if isinstance(data['RelatedTopics'][0], str) else data['RelatedTopics'][0].get('Text')}")

test_search("What is the latest score for the Lakers game?")
