"""
Benchmark Runner: Controlled execution of vector database benchmarks.
Applies identical embeddings, queries, warm-up, and evaluation metrics across all 8 systems.
"""
import os
import json
import time
import numpy as np
from typing import Dict, List, Any, Optional
from src.adapters import ADAPTER_REGISTRY, get_adapter, BaseVectorDBAdapter
from src.benchmark.metrics import (
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_hit_rate_at_k,
    calculate_mrr,
    calculate_ndcg_at_k,
    calculate_latency_statistics
)
from src.benchmark.resource_monitor import ResourceMonitor
from src.utils.logging_utils import setup_logger

logger = setup_logger("benchmark_runner")

class BenchmarkRunner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bench_cfg = config.get("benchmark", {})
        self.top_k = self.bench_cfg.get("top_k", 10)
        self.warmup_queries = self.bench_cfg.get("warmup_queries", 10)
        self.repetitions = self.bench_cfg.get("repetitions", 3)
        self.output_raw_dir = config.get("output", {}).get("raw_dir", "outputs/raw_results")

    def load_benchmark_assets(self) -> Dict[str, Any]:
        """Loads corpus, queries, qrels, and pre-computed embeddings."""
        proc_dir = self.config["dataset"]["processed_dir"]
        emb_dir = self.config["embedding"]["cache_dir"]

        with open(os.path.join(proc_dir, "corpus_subset.json"), "r", encoding="utf-8") as f:
            corpus = json.load(f)
        with open(os.path.join(proc_dir, "queries_subset.json"), "r", encoding="utf-8") as f:
            queries = json.load(f)
        with open(os.path.join(proc_dir, "qrels_subset.json"), "r", encoding="utf-8") as f:
            qrels = json.load(f)

        doc_embeddings = np.load(os.path.join(emb_dir, "doc_embeddings_384d.npy"))
        query_embeddings = np.load(os.path.join(emb_dir, "query_embeddings_384d.npy"))
        with open(os.path.join(emb_dir, "doc_ids.json"), "r", encoding="utf-8") as f:
            doc_ids = json.load(f)
        with open(os.path.join(emb_dir, "query_ids.json"), "r", encoding="utf-8") as f:
            query_ids = json.load(f)

        return {
            "corpus": corpus,
            "queries": queries,
            "qrels": qrels,
            "doc_embeddings": doc_embeddings,
            "query_embeddings": query_embeddings,
            "doc_ids": doc_ids,
            "query_ids": query_ids
        }

    def benchmark_database(self, db_key: str, assets: Dict[str, Any]) -> Dict[str, Any]:
        """Runs controlled benchmark on a single vector database adapter."""
        logger.info(f"\n==================================================")
        logger.info(f"STARTING BENCHMARK: {db_key.upper()}")
        logger.info(f"==================================================")

        adapter: BaseVectorDBAdapter = get_adapter(db_key, self.config)
        health = adapter.health_check()

        result = {
            "db_key": db_key,
            "name": adapter.name,
            "category": self.config.get("databases", {}).get(db_key, {}).get("category", "Vector DB"),
            "deployment_mode": health.deployment_mode,
            "version": health.version,
            "benchmarked": False,
            "status_reason": health.status_message,
            "ingestion": {},
            "latency": {},
            "throughput": {},
            "resources": {},
            "retrieval_quality": {}
        }

        if not health.is_available:
            logger.warning(f"[{adapter.name}] Skipping live benchmark: {health.status_message}")
            result["status_reason"] = f"Not Benchmarked: {health.status_message}"
            return result

        doc_embeddings = assets["doc_embeddings"]
        query_embeddings = assets["query_embeddings"]
        doc_ids = assets["doc_ids"]
        query_ids = assets["query_ids"]
        corpus = assets["corpus"]
        qrels = assets["qrels"]

        monitor = ResourceMonitor()
        monitor.start()

        collection_name = f"scifact_bench_{db_key}"

        try:
            # 1. Collection Creation
            logger.info(f"[{adapter.name}] Creating collection '{collection_name}' (dim=384, metric=cosine)...")
            adapter.create_collection(collection_name, dimension=384, metric="cosine")

            # 2. Ingestion
            logger.info(f"[{adapter.name}] Ingesting {len(doc_ids):,} vectors (batch_size=256)...")
            payloads = [{"title": corpus[did].get("title", "")} for did in doc_ids]
            
            t_insert = adapter.insert(doc_embeddings, payloads, doc_ids)
            vecs_per_sec = len(doc_ids) / t_insert if t_insert > 0 else 0.0
            logger.info(f"[{adapter.name}] Ingested in {t_insert:.3f}s ({vecs_per_sec:,.1f} vectors/sec)")

            # 3. Dedicated Index Build
            logger.info(f"[{adapter.name}] Building index / optimizing...")
            t_index = adapter.build_index()
            logger.info(f"[{adapter.name}] Index build time: {t_index:.3f}s")

            monitor.sample()
            storage_bytes = adapter.get_storage_size()
            storage_mb = round(storage_bytes / (1024 * 1024), 2) if storage_bytes else None

            # 4. Cache Warm-Up
            logger.info(f"[{adapter.name}] Executing {self.warmup_queries} warm-up queries...")
            warmup_vecs = query_embeddings[:self.warmup_queries]
            adapter.warmup(warmup_vecs, top_k=self.top_k)

            # 5. Timed Workload Execution
            logger.info(f"[{adapter.name}] Running {len(query_ids)} queries x {self.repetitions} repetitions (Top-K={self.top_k})...")
            all_latencies_ms = []
            per_query_results = []
            first_rep_retrieved = {} # query_id -> list of retrieved doc_ids

            t_total_query_start = time.perf_counter()

            for rep in range(self.repetitions):
                for q_idx, qid in enumerate(query_ids):
                    q_vec = query_embeddings[q_idx]

                    t0 = time.perf_counter_ns()
                    search_res = adapter.search(q_vec, top_k=self.top_k)
                    t_ns = time.perf_counter_ns() - t0
                    lat_ms = t_ns / 1_000_000.0

                    all_latencies_ms.append(lat_ms)
                    retrieved_ids = [r.doc_id for r in search_res]

                    if rep == 0:
                        first_rep_retrieved[qid] = retrieved_ids

                    per_query_results.append({
                        "database": adapter.name,
                        "repetition": rep + 1,
                        "query_idx": q_idx + 1,
                        "query_id": qid,
                        "latency_ms": lat_ms,
                        "num_retrieved": len(retrieved_ids),
                        "top_hit_id": retrieved_ids[0] if retrieved_ids else ""
                    })

            t_total_query_wall = time.perf_counter() - t_total_query_start
            total_queries_executed = len(all_latencies_ms)
            qps = total_queries_executed / t_total_query_wall if t_total_query_wall > 0 else 0.0

            # 6. Information Retrieval (IR) Evaluation using BEIR SciFact QRELS
            logger.info(f"[{adapter.name}] Calculating IR metrics against SciFact ground truth...")
            precisions, recalls, hit_rates, mrrs, ndcgs = [], [], [], [], []

            for qid in query_ids:
                retrieved_ids = first_rep_retrieved.get(qid, [])
                qrel_dict = qrels.get(qid, {})
                rel_set = {did for did, score in qrel_dict.items() if score > 0}

                p_k = calculate_precision_at_k(retrieved_ids, rel_set, k=self.top_k)
                r_k = calculate_recall_at_k(retrieved_ids, rel_set, k=self.top_k)
                hr_k = calculate_hit_rate_at_k(retrieved_ids, rel_set, k=self.top_k)
                mrr = calculate_mrr(retrieved_ids, rel_set)
                ndcg = calculate_ndcg_at_k(retrieved_ids, qrel_dict, k=self.top_k)

                precisions.append(p_k)
                recalls.append(r_k)
                hit_rates.append(hr_k)
                mrrs.append(mrr)
                ndcgs.append(ndcg)

            # 7. Aggregate Statistics
            lat_stats = calculate_latency_statistics(all_latencies_ms)
            res_summary = monitor.get_summary()

            result["benchmarked"] = True
            result["status_reason"] = "Successfully Benchmarked (Genuine Measured Run)"
            result["ingestion"] = {
                "total_insertion_time_sec": round(t_insert, 3),
                "vectors_per_sec": round(vecs_per_sec, 1),
                "index_build_time_sec": round(t_index, 3)
            }
            result["latency"] = {
                "mean_ms": round(lat_stats["mean"], 3),
                "p50_ms": round(lat_stats["p50"], 3),
                "p90_ms": round(lat_stats["p90"], 3),
                "p95_ms": round(lat_stats["p95"], 3),
                "p99_ms": round(lat_stats["p99"], 3),
                "min_ms": round(lat_stats["min"], 3),
                "max_ms": round(lat_stats["max"], 3),
                "std_ms": round(lat_stats["std"], 3)
            }
            result["throughput"] = {
                "qps": round(qps, 1),
                "total_queries_measured": total_queries_executed
            }
            result["resources"] = {
                "delta_rss_mb": res_summary["delta_rss_mb"],
                "peak_rss_mb": res_summary["peak_rss_mb"],
                "storage_mb": storage_mb
            }
            result["retrieval_quality"] = {
                "precision_at_10": round(float(np.mean(precisions)), 4),
                "recall_at_10": round(float(np.mean(recalls)), 4),
                "hit_rate_at_10": round(float(np.mean(hit_rates)), 4),
                "mrr": round(float(np.mean(mrrs)), 4),
                "ndcg_at_10": round(float(np.mean(ndcgs)), 4)
            }

            logger.info(f"[{adapter.name}] BENCHMARK COMPLETE:")
            logger.info(f"  - Ingestion: {vecs_per_sec:,.1f} vec/s | Index Build: {t_index:.2f}s")
            logger.info(f"  - Latency: P50={lat_stats['p50']:.2f}ms | P95={lat_stats['p95']:.2f}ms | P99={lat_stats['p99']:.2f}ms | Mean={lat_stats['mean']:.2f}ms")
            logger.info(f"  - Throughput: {qps:,.1f} QPS")
            logger.info(f"  - IR Quality: Recall@10={np.mean(recalls):.4f} | MRR={np.mean(mrrs):.4f} | NDCG@10={np.mean(ndcgs):.4f}")

            # Cleanup
            adapter.cleanup()

        except Exception as e:
            logger.error(f"[{adapter.name}] Benchmark failed during execution: {e}")
            result["benchmarked"] = False
            result["status_reason"] = f"Execution Error: {e}"
            try:
                adapter.cleanup()
            except Exception:
                pass

        return result

    def run_all(self) -> Dict[str, Any]:
        """Runs the entire benchmark suite over all 8 vector database systems."""
        assets = self.load_benchmark_assets()
        all_results = {}

        for db_key in ADAPTER_REGISTRY.keys():
            res = self.benchmark_database(db_key, assets)
            all_results[db_key] = res

        # Save raw JSON results
        os.makedirs(self.output_raw_dir, exist_ok=True)
        out_json = os.path.join(self.output_raw_dir, "benchmark_summary.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

        logger.info(f"\nAll raw results serialized to: {out_json}")
        return all_results
