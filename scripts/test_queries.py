import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.assistant.vdb_assistant import VDBAssistant

def run_tests():
    a = VDBAssistant()
    queries = [
        "Which has the lowest latency?",
        "Which has the highest QPS?",
        "Compare Qdrant and Chroma.",
        "Compare Qdrant and FAISS.",
        "Which has the best Recall@10?",
        "Which supports PostgreSQL?",
        "Which systems support metadata filtering?",
        "Which is suitable for a RAG application?",
        "What is the difference between Qdrant and Pinecone?",
        "Which is better overall?"
    ]

    for idx, q in enumerate(queries, 1):
        r = a.ask(q)
        print("=" * 80)
        print(f"TEST {idx}: {q}")
        print(f"INTENT: {r.get('intent')} | EVIDENCE: {r.get('evidence_type')}")
        print("-" * 80)
        print(r.get("answer"))
        print("\n")

if __name__ == "__main__":
    run_tests()
