"""
Evaluate complete RAG answers using LLM as judge
Optimized setup:
- Groq with Llama 3.1 8B (imported from query.py for RAG)
- Cerebras with llama3.3-70b (for HIGH QUALITY evaluation)
"""
import os
import sys
from dotenv import load_dotenv
from langfuse import Langfuse
import uuid

# Fix import path - add parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import from your existing query.py
from backend.query import load_vectordb, get_llm, create_rag_chain

load_dotenv()

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")


def evaluate_answer_quality(answer: str, question: str, expected: str = None):
    """
    Use Cerebras (llama3.3-70b) as judge to score answer quality
    
    Args:
        answer: The generated answer to evaluate
        question: The original question
        expected: (Optional) Expected/ground truth answer for comparison
    
    Returns:
        dict: Scores for correctness, completeness, relevance, and overall
    """
    import requests
    
    # ========== TWO DIFFERENT PROMPTS ==========
    
    if expected:
        # BATCH MODE: Compare against expected answer
        prompt = f"""You are evaluating RAG system answer quality.

Question: {question}

Expected Answer: {expected}

Actual Answer: {answer}

Rate the actual answer on these criteria (0.0 to 1.0 for each):
1. CORRECTNESS: Are the facts accurate compared to the expected answer?
2. COMPLETENESS: Does it answer all parts of the question?
3. RELEVANCE: Is it focused on the question?

Return ONLY three numbers separated by commas (correctness, completeness, relevance).
Example: 0.9, 0.8, 1.0"""
    
    else:
        # LIVE MODE: Evaluate without ground truth
        prompt = f"""You are evaluating a RAG system's answer quality WITHOUT a ground truth reference.

Question: {question}

Answer: {answer}

Rate the answer on these criteria (0.0 to 1.0 for each):
1. CORRECTNESS: Does the answer appear factually sound and coherent? (Check for logical consistency, no contradictions)
2. COMPLETENESS: Does it thoroughly address all aspects of the question?
3. RELEVANCE: Is the answer directly related to what was asked?

Return ONLY three numbers separated by commas (correctness, completeness, relevance).
Example: 0.9, 0.8, 1.0"""

    try:
        response = requests.post(
            "https://api.cerebras.ai/v1/chat/completions",  
            headers={
                "Authorization": f"Bearer {CEREBRAS_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama3.3-70b",  
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 50,
                "stream": False
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
    
    # Fallback scores
    return {"correctness": 0.5, "completeness": 0.5, "relevance": 0.5, "overall": 0.5}


def evaluate_single_live_answer(question: str, answer: str):
    """
    NEW FUNCTION: Evaluate a single answer in real-time (no expected answer needed)
    This is for live chat evaluation!
    
    Args:
        question: User's question
        answer: RAG system's answer
    
    Returns:
        dict: Evaluation scores
    """
    print(f"\n🔍 Evaluating live answer...")
    print(f"Question: {question[:60]}...")
    
    scores = evaluate_answer_quality(
        answer=answer,
        question=question,
        expected=None  # No ground truth for live evaluation
    )
    
    print(f"✓ Evaluation complete!")
    print(f"  Correctness:  {scores['correctness']:.2f}")
    print(f"  Completeness: {scores['completeness']:.2f}")
    print(f"  Relevance:    {scores['relevance']:.2f}")
    print(f"  Overall:      {scores['overall']:.2f}\n")
    
    return scores


def run_answer_evaluation(dataset_name: str, collection_name: str, k: int = 5):
    """
    BATCH EVALUATION: Evaluate complete RAG pipeline with automatic Langfuse logging
    Uses a dataset with expected answers
    """
    print("=" * 70)
    print("RAG ANSWER QUALITY EVALUATION (BATCH MODE)")
    print("=" * 70)
    print()
    
    langfuse = Langfuse()
    try:
        dataset = langfuse.get_dataset(name=dataset_name)
        items = dataset.items
        print(f"Dataset: {len(items)} questions")
        print(f"Using: k={k}")
        print(f"Logging to Langfuse automatically...")
        print()
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Reuse components from query.py (Groq for generation)
    vectordb = load_vectordb(collection_name)
    llm = get_llm()
    rag_chain, retriever = create_rag_chain(vectordb, llm, k=k)
    
    # Evaluate each question
    all_scores = []
    
    for i, item in enumerate(items, 1):
        question = str(item.input).strip()
        expected = str(item.expected_output).strip()
        
        print(f"Q{i:2d}: {question[:60]}...")
        
        try:
            # Create a unique trace_id for this evaluation
            trace_id = f"eval_{dataset_name}_{i}_{str(uuid.uuid4())[:8]}"
            
            # Generate answer
            answer = rag_chain.invoke(question)
            
            # Evaluate (with expected answer for comparison)
            scores = evaluate_answer_quality(
                answer=answer,
                question=question,
                expected=expected  # Batch mode has ground truth
            )
            all_scores.append(scores)
            
            # ========== LOG SCORES TO LANGFUSE ==========
            try:
                langfuse.create_score(
                    trace_id=trace_id,
                    name="correctness",
                    value=scores["correctness"],
                    data_type="NUMERIC",
                    comment=f"Q{i}: {question[:40]}..."
                )
                
                langfuse.create_score(
                    trace_id=trace_id,
                    name="completeness",
                    value=scores["completeness"],
                    data_type="NUMERIC",
                    comment=f"Q{i}: Answer completeness"
                )
                
                langfuse.create_score(
                    trace_id=trace_id,
                    name="relevance",
                    value=scores["relevance"],
                    data_type="NUMERIC",
                    comment=f"Q{i}: Question relevance"
                )
                
                langfuse.create_score(
                    trace_id=trace_id,
                    name="overall_quality",
                    value=scores["overall"],
                    data_type="NUMERIC",
                    comment=f"Q{i}: Overall evaluation score"
                )
                
                print(f"     Correctness: {scores['correctness']:.2f}")
                print(f"     Completeness: {scores['completeness']:.2f}")
                print(f"     Relevance: {scores['relevance']:.2f}")
                print(f"     Overall: {scores['overall']:.2f}")
                print(f"     ✓ Logged to Langfuse (trace: {trace_id})")
                print()
                
            except Exception as score_error:
                print(f"     Correctness: {scores['correctness']:.2f}")
                print(f"     Completeness: {scores['completeness']:.2f}")
                print(f"     Relevance: {scores['relevance']:.2f}")
                print(f"     Overall: {scores['overall']:.2f}")
                print(f"     ⚠ Langfuse logging failed: {score_error}")
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
    
    # Log summary scores
    try:
        summary_trace_id = f"eval_summary_{dataset_name}_{str(uuid.uuid4())[:8]}"
        
        langfuse.create_score(
            trace_id=summary_trace_id,
            name="avg_correctness",
            value=avg_correctness,
            data_type="NUMERIC",
            comment=f"Average across {len(items)} questions - {dataset_name}"
        )
        
        langfuse.create_score(
            trace_id=summary_trace_id,
            name="avg_completeness",
            value=avg_completeness,
            data_type="NUMERIC",
            comment=f"Average across {len(items)} questions"
        )
        
        langfuse.create_score(
            trace_id=summary_trace_id,
            name="avg_relevance",
            value=avg_relevance,
            data_type="NUMERIC",
            comment=f"Average across {len(items)} questions"
        )
        
        langfuse.create_score(
            trace_id=summary_trace_id,
            name="avg_overall",
            value=avg_overall,
            data_type="NUMERIC",
            comment=f"Overall evaluation: {dataset_name} (k={k})"
        )
    except Exception as e:
        print(f"\n⚠ Could not log summary to Langfuse: {e}")
    
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
    
    # Flush to ensure all data is sent
    langfuse.flush()
    print("\n✓ All scores flushed to Langfuse")
    
    return all_scores


if __name__ == "__main__":
    print("\nRAG Answer Quality Evaluation\n")
    
    # Ask user which mode they want
    mode = input("Mode? (1) Batch evaluation with dataset, (2) Single live evaluation: ").strip()
    
    if mode == "2":
        # LIVE MODE
        print("\n=== Live Answer Evaluation ===\n")
        question = input("Enter question: ").strip()
        answer = input("Enter answer: ").strip()
        
        if question and answer:
            scores = evaluate_single_live_answer(question, answer)
            print("\nDone! This answer can be evaluated without a dataset.")
        else:
            print("Error: Both question and answer are required!")
    
    else:
        # BATCH MODE (original functionality)
        dataset_name = input("Dataset name: ").strip()
        collection_name = input("Collection name: ").strip()
        
        if not collection_name:
            print("Error: Collection name is required!")
            exit(1)
            
        k = int(input("k-value (default: 5): ").strip() or "5")
        
        print()
        run_answer_evaluation(dataset_name, collection_name, k)
