"""
Script to execute the master benchmark across all 8 vector database systems.
Uses identical SciFact 1,000-doc / 50-query dataset, 384-d MiniLM embeddings, Top-K=10.
"""
import os
import sys
import json
import yaml

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.benchmark.runner import BenchmarkRunner
from src.utils.logging_utils import setup_logger

logger = setup_logger("run_benchmark")

def main():
    config_path = "config/benchmark_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("=" * 70)
    logger.info("PHASE 5: MASTER BENCHMARK EXECUTION (8 VECTOR DATABASES)")
    logger.info("=" * 70)

    runner = BenchmarkRunner(config)
    results = runner.run_all()

    print("\n" + "=" * 90)
    print("MASTER BENCHMARK EXECUTION SUMMARY (1,000 DOCS / 50 QUERIES / TOP-K=10)")
    print("=" * 90)
    print(f"{'VDB Name':<12} | {'Status':<16} | {'Vec/Sec':<10} | {'P50 (ms)':<10} | {'P95 (ms)':<10} | {'QPS':<8} | {'Recall@10':<10} | {'NDCG@10':<10}")
    print("-" * 90)

    for db_key, res in results.items():
        name = res["name"]
        if res["benchmarked"]:
            status = "BENCHMARKED"
            vecs = f"{res['ingestion'].get('vectors_per_sec', 0):,.1f}"
            p50 = f"{res['latency'].get('p50_ms', 0):.2f}"
            p95 = f"{res['latency'].get('p95_ms', 0):.2f}"
            qps = f"{res['throughput'].get('qps', 0):,.1f}"
            rec = f"{res['retrieval_quality'].get('recall_at_10', 0):.4f}"
            ndcg = f"{res['retrieval_quality'].get('ndcg_at_10', 0):.4f}"
            print(f"{name:<12} | {status:<16} | {vecs:<10} | {p50:<10} | {p95:<10} | {qps:<8} | {rec:<10} | {ndcg:<10}")
        else:
            status = "NOT BENCHMARKED"
            print(f"{name:<12} | {status:<16} | {'-':<10} | {'-':<10} | {'-':<10} | {'-':<8} | {'-':<10} | {'-':<10}")
            print(f"  -> Reason: {res['status_reason']}")

    print("-" * 90)
    print("All experimental results stored in: outputs/raw_results/benchmark_summary.json")
    print("=" * 90 + "\n")

if __name__ == "__main__":
    main()
