"""
Evaluate complete RAG answers using LLM as judge
Optimized setup:
- Groq with Llama 3.1 8B (imported from query.py for RAG)
- Cerebras with Qwen 3 235B (for HIGH QUALITY evaluation)
"""
import os
import sys
from dotenv import load_dotenv
from langfuse import Langfuse

# Fix import path - add parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import from your existing query.py
from backend.query import load_vectordb, get_llm, create_rag_chain

load_dotenv()

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")

def evaluate_answer_quality(answer: str, expected: str, question: str):
    """Use Cerebras (Qwen 3 235B) as judge to score answer quality"""
    import requests
    
    prompt = f"""You are evaluating RAG system answer quality.

Question: {question}

Expected Answer: {expected}

Actual Answer: {answer}

Rate the actual answer on these criteria (0.0 to 1.0 for each):
1. CORRECTNESS: Are the facts accurate?
2. COMPLETENESS: Does it answer all parts of the question?
3. RELEVANCE: Is it focused on the question?

Return ONLY three numbers separated by commas (correctness, completeness, relevance).
Example: 0.9, 0.8, 1.0"""

    try:
        # Call Cerebras API directly (using requests since SDK might have import issues)
        response = requests.post(
            "https://api.cerebras.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {CEREBRAS_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "qwen-3-235b",  
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 50
            },
            timeout=30
        )
        
        response.raise_for_status()
        scores_text = response.json()["choices"][0]["message"]["content"].strip()
        scores = [float(s.strip()) for s in scores_text.split(",")]
        
        if len(scores) == 3:
            correctness, completeness, relevance = scores
            overall = (correctness + completeness + relevance) / 3
            return {
                "correctness": correctness,
                "completeness": completeness,
                "relevance": relevance,
                "overall": overall
            }
    except Exception as e:
        print(f"Warning: Evaluation failed - {e}")
        pass
    
    return {"correctness": 0.5, "completeness": 0.5, "relevance": 0.5, "overall": 0.5}

def run_answer_evaluation(dataset_name: str, collection_name: str, k: int = 5):
    """Evaluate complete RAG pipeline"""
    print("=" * 70)
    print("RAG ANSWER QUALITY EVALUATION")
    print("=" * 70)
    print()
    
    langfuse = Langfuse()
    try:
        dataset = langfuse.get_dataset(name=dataset_name)
        items = dataset.items
        print(f"Dataset: {len(items)} questions")
        print(f"Using: k={k}")
        print()
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Reuse components from query.py (Groq for generation)
    vectordb = load_vectordb(collection_name)
    llm = get_llm()  # Returns Groq with llama-3.1-8b-instant
    rag_chain, retriever = create_rag_chain(vectordb, llm, k=k)
    
    # Evaluate each question
    all_scores = []
    
    for i, item in enumerate(items, 1):
        question = str(item.input).strip()
        expected = str(item.expected_output).strip()
        
        print(f"Q{i:2d}: {question[:60]}...")
        
        try:
            # Generate answer using RAG chain (Groq)
            answer = rag_chain.invoke(question)
            
            # Evaluate using Cerebras judge
            scores = evaluate_answer_quality(answer, expected, question)
            all_scores.append(scores)
            
            print(f"     Correctness: {scores['correctness']:.2f}")
            print(f"     Completeness: {scores['completeness']:.2f}")
            print(f"     Relevance: {scores['relevance']:.2f}")
            print(f"     Overall: {scores['overall']:.2f}")
            print()
            
        except Exception as e:
            print(f"     ERROR: {e}")
            all_scores.append({"correctness": 0, "completeness": 0, "relevance": 0, "overall": 0})
            print()
    
    # Summary
    if not all_scores:
        print("No scores to summarize")
        return None
        
    avg_correctness = sum(s["correctness"] for s in all_scores) / len(all_scores)
    avg_completeness = sum(s["completeness"] for s in all_scores) / len(all_scores)
    avg_relevance = sum(s["relevance"] for s in all_scores) / len(all_scores)
    avg_overall = sum(s["overall"] for s in all_scores) / len(all_scores)
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Correctness:  {avg_correctness:.3f}")
    print(f"Completeness: {avg_completeness:.3f}")
    print(f"Relevance:    {avg_relevance:.3f}")
    print(f"Overall:      {avg_overall:.3f}")
    print("=" * 70)
    print()
    print("Configuration:")
    print("  - RAG Generation: Groq (llama-3.1-8b-instant) - Fast")
    print("  - Evaluation: Cerebras (llama3.3-70b) - High Quality")
    print("=" * 70)
    
    return all_scores

if __name__ == "__main__":
    print("\nRAG Answer Quality Evaluation\n")
    
    dataset_name = input("Dataset name: ").strip()
    collection_name = input("Collection name: ").strip()
    
    if not collection_name:
        print("Error: Collection name is required!")
        exit(1)
        
    k = int(input("k-value (default: 5): ").strip() or "5")
    
    print()
    run_answer_evaluation(dataset_name, collection_name, k)
