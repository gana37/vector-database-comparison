"""
Structured Data Loader and Exporter for Vector Database Comparison.
Provides a unified data layer so that Excel, Dashboard, and Assistant
all operate on the exact same underlying facts and benchmark results.
"""
import os
import json
from typing import Dict, List, Any, Optional

from src.analysis.comparison import get_exhaustive_technical_data, DATABASES

DEFAULT_SUMMARY_PATH = "outputs/raw_results/benchmark_summary.json"
DEFAULT_JSON_PATH = "data/vdb_comparison.json"

WORKLOAD_METADATA = {
    "dataset": "BEIR/SciFact",
    "dataset_source": "TU Darmstadt (Thakur et al., 2021)",
    "documents_count": 1000,
    "queries_count": 50,
    "relevance_judgments_count": 54,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "embedding_dimensions": 384,
    "normalization": "L2 Unit Normalization (||v||_2 = 1.0)",
    "similarity_metric": "Cosine Similarity / Inner Product",
    "top_k": 10,
    "warmup_queries": 10,
    "benchmark_repetitions": 3,
    "total_queries_measured": 150,
    "timing_mechanism": "time.perf_counter_ns (Monotonic High-Precision)",
    "faiss_scope_notice": (
        "FAISS is an in-process vector similarity-search C++ library and algorithmic compute baseline, "
        "not a full client-server vector database. It lacks a network daemon, durability (WAL), "
        "CRUD mutations, metadata payload filtering, and distributed clustering."
    ),
    "workload_scope_notice": (
        "The empirical benchmark results apply strictly to this controlled workload (1,000 documents, "
        "50 queries, 384 dimensions, local execution) and should not be treated as universal performance claims."
    )
}

SEARCH_TYPE_MAP = {
    "qdrant": "Approximate Nearest Neighbor (HNSW, M=16, ef_construct=100, Embedded In-Memory)",
    "chroma": "Approximate Nearest Neighbor (HNSW via hnswlib, Persistent Disk Storage)",
    "faiss": "Exact Nearest Neighbor Compute Baseline (IndexFlatIP, Pure In-Memory)",
    "milvus": "Approximate Nearest Neighbor (HNSW, Distributed Cluster)",
    "weaviate": "Approximate Nearest Neighbor (Dynamic HNSW, LSM-tree)",
    "pgvector": "Approximate Nearest Neighbor (HNSW / IVFFlat, PostgreSQL Extension)",
    "elasticsearch": "Approximate Nearest Neighbor (Lucene HNSW, Distributed Cluster)",
    "pinecone": "Approximate Nearest Neighbor (Proprietary Graph, Cloud Serverless SaaS)"
}

DATABASE_NAME_TO_KEY = {
    "Qdrant": "qdrant",
    "Chroma": "chroma",
    "FAISS": "faiss",
    "Milvus": "milvus",
    "Weaviate": "weaviate",
    "pgvector": "pgvector",
    "Elasticsearch": "elasticsearch",
    "Pinecone": "pinecone"
}


def build_consolidated_vdb_dataset(
    summary_path: str = DEFAULT_SUMMARY_PATH
) -> Dict[str, Any]:
    """
    Builds the unified dataset combining empirical benchmark measurements
    and documented technical matrix across all 8 vector databases.
    """
    benchmark_summary: Dict[str, Any] = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            benchmark_summary = json.load(f)

    # Exhaustive technical matrix rows
    raw_matrix = get_exhaustive_technical_data(benchmark_summary)

    benchmarked_keys = ["qdrant", "chroma", "faiss"]
    unbenchmarked_keys = ["milvus", "weaviate", "pgvector", "elasticsearch", "pinecone"]

    measured_results = {}
    for db_key in benchmarked_keys:
        raw_db = benchmark_summary.get(db_key, {})
        measured_results[db_key] = {
            "name": raw_db.get("name", db_key.capitalize()),
            "category": raw_db.get("category", ""),
            "deployment_mode": raw_db.get("deployment_mode", ""),
            "version": raw_db.get("version", ""),
            "search_type": SEARCH_TYPE_MAP.get(db_key, ""),
            "benchmarked": True,
            "status_reason": raw_db.get("status_reason", "Successfully Benchmarked"),
            "ingestion": {
                "vectors_per_sec": raw_db.get("ingestion", {}).get("vectors_per_sec", 0.0),
                "total_insertion_time_sec": raw_db.get("ingestion", {}).get("total_insertion_time_sec", 0.0),
                "index_build_time_sec": raw_db.get("ingestion", {}).get("index_build_time_sec", 0.0)
            },
            "latency_ms": {
                "p50": raw_db.get("latency", {}).get("p50_ms", 0.0),
                "p90": raw_db.get("latency", {}).get("p90_ms", 0.0),
                "p95": raw_db.get("latency", {}).get("p95_ms", 0.0),
                "p99": raw_db.get("latency", {}).get("p99_ms", 0.0),
                "mean": raw_db.get("latency", {}).get("mean_ms", 0.0),
                "min": raw_db.get("latency", {}).get("min_ms", 0.0),
                "max": raw_db.get("latency", {}).get("max_ms", 0.0),
                "std": raw_db.get("latency", {}).get("std_ms", 0.0)
            },
            "throughput_qps": raw_db.get("throughput", {}).get("qps", 0.0),
            "retrieval_quality": {
                "precision_at_10": raw_db.get("retrieval_quality", {}).get("precision_at_10", 0.0),
                "recall_at_10": raw_db.get("retrieval_quality", {}).get("recall_at_10", 0.0),
                "hit_rate_at_10": raw_db.get("retrieval_quality", {}).get("hit_rate_at_10", 0.0),
                "mrr": raw_db.get("retrieval_quality", {}).get("mrr", 0.0),
                "ndcg_at_10": raw_db.get("retrieval_quality", {}).get("ndcg_at_10", 0.0)
            },
            "resources": {
                "peak_rss_mb": raw_db.get("resources", {}).get("peak_rss_mb"),
                "delta_rss_mb": raw_db.get("resources", {}).get("delta_rss_mb"),
                "storage_mb": raw_db.get("resources", {}).get("storage_mb")
            }
        }

    unbenchmarked_details = {}
    for db_key in unbenchmarked_keys:
        raw_db = benchmark_summary.get(db_key, {})
        unbenchmarked_details[db_key] = {
            "name": raw_db.get("name", db_key.capitalize()),
            "category": raw_db.get("category", ""),
            "deployment_mode": raw_db.get("deployment_mode", ""),
            "benchmarked": False,
            "status_reason": raw_db.get("status_reason", "Not Benchmarked"),
            "search_type": SEARCH_TYPE_MAP.get(db_key, "Documented Capability"),
            "reason_summary": _extract_short_reason(raw_db.get("status_reason", ""))
        }

    # Build per-database profile dictionary from the matrix
    database_profiles = _build_database_profiles(raw_matrix)

    consolidated = {
        "metadata": WORKLOAD_METADATA,
        "all_databases": DATABASES,
        "benchmarked_systems": [DATABASE_NAME_TO_KEY[name] for name in ["Qdrant", "Chroma", "FAISS"]],
        "unbenchmarked_systems": [DATABASE_NAME_TO_KEY[name] for name in ["Milvus", "Weaviate", "pgvector", "Elasticsearch", "Pinecone"]],
        "measured_results": measured_results,
        "unbenchmarked_details": unbenchmarked_details,
        "database_profiles": database_profiles,
        "technical_matrix_rows": raw_matrix
    }

    return consolidated


def _extract_short_reason(full_reason: str) -> str:
    if "pymilvus" in full_reason:
        return "pymilvus client uninstalled (requires running Docker container/cluster)"
    elif "weaviate" in full_reason:
        return "weaviate-client uninstalled (requires running Weaviate daemon)"
    elif "psycopg2" in full_reason:
        return "psycopg2 uninstalled (requires PostgreSQL server with pgvector extension)"
    elif "elasticsearch" in full_reason:
        return "elasticsearch client uninstalled (requires active Elasticsearch cluster daemon)"
    elif "PINECONE_API_KEY" in full_reason:
        return "PINECONE_API_KEY unconfigured (Cloud SaaS skipped to avoid unauthorized access/billing)"
    return full_reason or "Not Benchmarked"


def _build_database_profiles(matrix_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Builds a key-value dictionary for each of the 8 databases from matrix rows."""
    profiles: Dict[str, Dict[str, Any]] = {
        name: {"name": name, "key": DATABASE_NAME_TO_KEY.get(name, name.lower()), "attributes": {}}
        for name in DATABASES
    }

    for row in matrix_rows:
        param = row.get("parameter", "")
        section = row.get("section", "")
        row_type = row.get("type", "DOCUMENTATION")

        for db_name in DATABASES:
            val = row.get(db_name, "")
            profiles[db_name]["attributes"][param] = {
                "value": val,
                "section": section,
                "type": row_type
            }

    return profiles


def export_structured_data(
    output_path: str = DEFAULT_JSON_PATH,
    summary_path: str = DEFAULT_SUMMARY_PATH
) -> str:
    """Exports consolidated JSON dataset to disk."""
    data = build_consolidated_vdb_dataset(summary_path=summary_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return output_path


def load_structured_data(filepath: str = DEFAULT_JSON_PATH) -> Dict[str, Any]:
    """Loads consolidated dataset, regenerating if absent."""
    if not os.path.exists(filepath):
        export_structured_data(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
