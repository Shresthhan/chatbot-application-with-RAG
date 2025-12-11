"""
Simple RAG Evaluation - No Langfuse complexity
Just calculates retrieval relevance scores and prints results
"""

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langfuse import Langfuse

load_dotenv()

# Configuration
CHROMA_PATH = "./Vector_DB"


def load_vectordb(collection_name: str):
    """Load vector database"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name=collection_name,
    )


def calculate_relevance(docs, expected_answer: str):
    """Calculate how many chunks are relevant based on word overlap"""
    if not docs:
        return 0.0, "No documents retrieved"

    expected_words = set(expected_answer.lower().split())
    if not expected_words:
        return 0.0, "Expected answer is empty"

    relevant_count = 0

    for doc in docs:
        doc_words = set(doc.page_content.lower().split())
        overlap = len(expected_words & doc_words)
        # Consider relevant if there is reasonable overlap
        if overlap >= 5:
            relevant_count += 1

    score = relevant_count / len(docs)
    return score, f"{relevant_count}/{len(docs)} chunks relevant"


def run_evaluation(dataset_name: str, collection_name: str):
    """Run evaluation across several k values and report averages"""
    print("=" * 70)
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 70)
    print()

    # Load dataset from Langfuse
    lf = Langfuse()
    try:
        dataset = lf.get_dataset(name=dataset_name)
        items = dataset.items
        print(f"Dataset loaded: {len(items)} questions")
        print()
    except Exception as e:
        print(f"Error loading dataset '{dataset_name}': {e}")
        return

    k_values = [3, 5, 7, 10]
    all_results = {}

    for k in k_values:
        print(f"Testing k={k}...")
        vectordb = load_vectordb(collection_name)
        retriever = vectordb.as_retriever(search_kwargs={"k": k})

        scores = []

        for i, item in enumerate(items, 1):
            question = str(item.input).strip()
            expected = str(item.expected_output).strip()

            try:
                docs = retriever.invoke(question)
                score, comment = calculate_relevance(docs, expected)
                scores.append(score)
                print(f"  Q{i:2d}: score={score:.2f} ({comment})")
            except Exception as e:
                print(f"  Q{i:2d}: ERROR - {e}")
                scores.append(0.0)

        avg = sum(scores) / len(scores) if scores else 0.0
        all_results[k] = {"scores": scores, "average": avg, "count": len(scores)}

        print(f"  Average score for k={k}: {avg:.3f}")
        print()

    # Summary
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()
    print(f"{'k-value':<10} {'Avg Score':<12} {'Visualization'}")
    print("-" * 70)

    best_avg = max(r["average"] for r in all_results.values()) if all_results else 0.0

    for k in k_values:
        avg = all_results[k]["average"]
        bar = "█" * int(avg * 50)
        marker = "*" if avg == best_avg else " "
        print(f"{marker} k={k:<5} | {avg:.3f}      | {bar}")

    best_k = max(all_results.items(), key=lambda x: x[1]["average"]) if all_results else (None, {"average": 0.0})

    print()
    print("=" * 70)
    print(f"Recommended k: {best_k[0]} (average score {best_k[1]['average']:.3f})")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Update frontend/app_api.py (retrieval_k default) to this k value.")
    print("  2. Restart the Streamlit app.")
    print("  3. Manually verify answer quality with this setting.")
    print()

    return all_results


if __name__ == "__main__":
    print("Simple RAG Retrieval Evaluation")
    print()

    dataset_name = input("Dataset name: ").strip()
    collection_name = input("Collection name (default: research_paper): ").strip() or "research_paper"

    print()
    run_evaluation(dataset_name, collection_name)