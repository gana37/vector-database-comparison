"""
CLI Interactive Assistant for Vector Database Comparison.
Provides an interactive command-line interface to the data-grounded assistant.
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.assistant.vdb_assistant import VDBAssistant

def main():
    assistant = VDBAssistant()
    print("=" * 75)
    print("VECTOR DATABASE COMPARATIVE ANALYSIS ASSISTANT (CLI MODE)")
    print("Data-Grounded in Empirical Benchmark & Documented Technical Architecture")
    print("Type 'exit' or 'quit' to terminate.")
    print("=" * 75)

    sample_questions = [
        "1. Which has the lowest latency?",
        "2. Which has the highest QPS?",
        "3. Compare Qdrant and Chroma.",
        "4. Compare Qdrant and FAISS.",
        "5. Which has the best Recall@10?",
        "6. Which supports PostgreSQL?",
        "7. Which systems support metadata filtering?",
        "8. Which is suitable for a RAG application?",
        "9. What is the difference between Qdrant and Pinecone?",
        "10. Which is better overall?"
    ]
    print("\nSample queries:")
    for sq in sample_questions:
        print(f"  {sq}")
    print("-" * 75)

    if len(sys.argv) > 1:
        # One-shot query mode
        query = " ".join(sys.argv[1:])
        print(f"\nUser Query: {query}\n")
        res = assistant.ask(query)
        print(res["answer"])
        return

    while True:
        try:
            query = input("\nAsk Assistant > ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("Exiting assistant. Goodbye!")
                break
            res = assistant.ask(query)
            print("\n" + "=" * 70)
            print(f"INTENT: {res['intent']} | EVIDENCE: [{res['evidence_type']}]")
            print("=" * 70)
            print(res["answer"])
            print("=" * 70)
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break

if __name__ == "__main__":
    main()
