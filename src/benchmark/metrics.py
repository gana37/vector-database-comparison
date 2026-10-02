"""
Evaluation Metrics for Vector Database Benchmarking.
Implements Information Retrieval (IR) metrics using official BEIR SciFact QRELS,
and statistical latency/throughput performance calculations.
"""
import math
import numpy as np
from typing import Dict, List, Any, Optional

def calculate_precision_at_k(retrieved_ids: List[str], relevant_ids: set, k: int = 10) -> float:
    """Precision@K = (Number of relevant docs in top-K) / K"""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_retrieved = sum(1 for did in top_k if did in relevant_ids)
    return relevant_retrieved / k

def calculate_recall_at_k(retrieved_ids: List[str], relevant_ids: set, k: int = 10) -> float:
    """Recall@K = (Number of relevant docs in top-K) / (Total relevant docs in subset)"""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_retrieved = sum(1 for did in top_k if did in relevant_ids)
    return relevant_retrieved / len(relevant_ids)

def calculate_hit_rate_at_k(retrieved_ids: List[str], relevant_ids: set, k: int = 10) -> float:
    """Hit Rate@K = 1.0 if at least one relevant doc is in top-K, else 0.0"""
    top_k = retrieved_ids[:k]
    return 1.0 if any(did in relevant_ids for did in top_k) else 0.0

def calculate_mrr(retrieved_ids: List[str], relevant_ids: set) -> float:
    """MRR = 1 / rank of the first relevant document (1-indexed)"""
    for rank, did in enumerate(retrieved_ids, start=1):
        if did in relevant_ids:
            return 1.0 / rank
    return 0.0

def calculate_ndcg_at_k(retrieved_ids: List[str], qrel_dict: Dict[str, int], k: int = 10) -> float:
    """
    NDCG@K = DCG@K / IDCG@K
    DCG@K = sum((2^rel - 1) / log2(rank + 1))
    """
    top_k = retrieved_ids[:k]
    dcg = 0.0
    for rank, did in enumerate(top_k, start=1):
        rel = qrel_dict.get(did, 0)
        if rel > 0:
            dcg += (math.pow(2, rel) - 1.0) / math.log2(rank + 1)

    # Ideal DCG: sort all relevant scores descending
    ideal_rels = sorted(list(qrel_dict.values()), reverse=True)[:k]
    idcg = 0.0
    for rank, rel in enumerate(ideal_rels, start=1):
        if rel > 0:
            idcg += (math.pow(2, rel) - 1.0) / math.log2(rank + 1)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg

def calculate_latency_statistics(latencies_ms: List[float]) -> Dict[str, float]:
    """Computes high-precision percentile distribution for query latencies."""
    if not latencies_ms:
        return {
            "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
            "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0
        }
    arr = np.array(latencies_ms, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }
