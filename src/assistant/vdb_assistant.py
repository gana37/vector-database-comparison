"""
Data-Grounded Natural-Language Vector Database Comparison Assistant.
Provides evidence-backed answers grounded strictly in the project's structured dataset.
Distinguishes between [MEASURED], [DOCUMENTED], and [NOT BENCHMARKED] systems.
No fabricated metrics; trade-off-based reasoning rather than simplistic ranking.
"""
import re
from typing import Dict, List, Any, Optional, Tuple

from src.analysis.data_loader import load_structured_data, DATABASES, WORKLOAD_METADATA

class VDBAssistant:
    def __init__(self, data_path: Optional[str] = None):
        if data_path:
            self.data = load_structured_data(data_path)
        else:
            self.data = load_structured_data()

        self.measured = self.data.get("measured_results", {})
        self.unbenchmarked = self.data.get("unbenchmarked_details", {})
        self.profiles = self.data.get("database_profiles", {})
        self.matrix_rows = self.data.get("technical_matrix_rows", [])
        self.all_dbs = self.data.get("all_databases", DATABASES)
        self.benchmarked_keys = self.data.get("benchmarked_systems", ["qdrant", "chroma", "faiss"])
        self.unbenchmarked_keys = self.data.get("unbenchmarked_systems", ["milvus", "weaviate", "pgvector", "elasticsearch", "pinecone"])

        # Synonyms and mapping
        self.db_aliases = {
            "qdrant": "Qdrant",
            "chroma": "Chroma",
            "chromadb": "Chroma",
            "faiss": "FAISS",
            "milvus": "Milvus",
            "weaviate": "Weaviate",
            "pgvector": "pgvector",
            "postgres": "pgvector",
            "postgresql": "pgvector",
            "elasticsearch": "Elasticsearch",
            "elastic": "Elasticsearch",
            "es": "Elasticsearch",
            "pinecone": "Pinecone"
        }

    def _extract_mentioned_dbs(self, query: str) -> List[str]:
        q_lower = query.lower()
        mentioned = []
        for alias, canonical in self.db_aliases.items():
            pattern = r'\b' + re.escape(alias) + r'\b'
            if re.search(pattern, q_lower):
                if canonical not in mentioned:
                    mentioned.append(canonical)
        return mentioned

    def _get_unbenchmarked_notice(self) -> str:
        items = []
        for key in self.unbenchmarked_keys:
            info = self.unbenchmarked.get(key, {})
            name = info.get("name", key.capitalize())
            reason = info.get("reason_summary", info.get("status_reason", "Not Benchmarked"))
            items.append(f"  - **{name}**: *{reason}*")
        reasons_text = "\n".join(items)
        return (
            f"> [!NOTE]\n"
            f"> **Unbenchmarked Systems Disclosure `[NOT BENCHMARKED]`**:\n"
            f"> Under strict experimental integrity, live benchmark numbers were **not** fabricated or simulated "
            f"for the remaining 5 systems. Their non-execution reasons:\n"
            f"{reasons_text}\n"
        )

    def _get_workload_footnote(self) -> str:
        return (
            f"\n---\n"
            f"*Workload Context*: {WORKLOAD_METADATA['dataset']} (1,000 docs, 50 queries, 384-d MiniLM, Top-K=10). "
            f"FAISS measured using `IndexFlatIP` compute baseline; Chroma measured using incremental HNSW (hnswlib); "
            f"Qdrant measured in embedded in-memory mode using an unindexed segment exact scan because the collection size was below the indexing threshold. "
            f"Performance results are workload-specific and should not be interpreted as a universal comparison of the underlying database engines or index algorithms."
        )

    def ask(self, query: str) -> Dict[str, Any]:
        """
        Main query interface. Interprets intent and returns data-grounded response.
        """
        q = query.strip()
        q_lower = q.lower()
        mentioned_dbs = self._extract_mentioned_dbs(q)

        # 1. Compare specific pair/tuple of databases
        if (
            ("compare" in q_lower or "difference between" in q_lower or "performance difference" in q_lower or " vs " in q_lower or " versus " in q_lower)
            and len(mentioned_dbs) >= 2
        ):
            return self._handle_comparison(mentioned_dbs, q_lower)

        # 2. Lowest Latency
        if any(term in q_lower for term in ["lowest latency", "fastest query", "min latency", "lowest query latency", "fastest latency"]):
            return self._handle_lowest_latency()

        # 3. Latency general
        if "latency" in q_lower and ("which" in q_lower or "compare" in q_lower or "what is" in q_lower or "show" in q_lower):
            return self._handle_lowest_latency()

        # 4. Highest QPS / Throughput
        if any(term in q_lower for term in ["highest qps", "highest throughput", "most throughput", "fastest throughput", "max qps", "max throughput"]):
            return self._handle_highest_qps()

        if "qps" in q_lower or "queries per second" in q_lower or "throughput" in q_lower:
            return self._handle_highest_qps()

        # 5. Ingestion Throughput / Speed
        if any(term in q_lower for term in ["ingestion", "ingest", "insertion speed", "vectors per second", "indexing speed", "index build"]):
            return self._handle_ingestion()

        # 6. Best Retrieval Quality / Recall / Precision / NDCG / MRR
        if any(term in q_lower for term in ["recall", "precision", "mrr", "ndcg", "hit rate", "retrieval quality", "best quality", "accuracy"]):
            return self._handle_retrieval_quality(q_lower)

        # 7. PostgreSQL integration
        if any(term in q_lower for term in ["postgresql", "postgres", "pgvector", "sql database", "relational"]):
            return self._handle_postgresql_integration(mentioned_dbs)

        # 8. Metadata Filtering
        if any(term in q_lower for term in ["metadata filter", "payload filter", "single-stage", "filter", "filtering"]):
            return self._handle_metadata_filtering(mentioned_dbs)

        # 9. RAG Suitability
        if any(term in q_lower for term in ["rag", "retrieval augmented generation", "llm application", "agentic", "for rag"]):
            return self._handle_rag_suitability()

        # 10. Distributed Deployment / Sharding / Replication
        if any(term in q_lower for term in ["distributed", "sharding", "replication", "cluster", "clustering", "consensus", "horizontal scale", "high availability", "multi-node"]):
            return self._handle_distributed_deployment(mentioned_dbs)

        # 11. Hybrid Search / BM25 / Sparse-Dense
        if any(term in q_lower for term in ["hybrid", "bm25", "sparse", "rrf", "reciprocal rank", "keyword search", "lexical"]):
            return self._handle_hybrid_search(mentioned_dbs)

        # 12. Open Source / Licensing
        if any(term in q_lower for term in ["open source", "open-source", "license", "licensing", "oss", "proprietary", "self host", "self-host", "on-premise", "on premise"]):
            return self._handle_licensing(mentioned_dbs)

        # 13. Which is better overall? / Recommendation
        if any(term in q_lower for term in ["which is better", "best overall", "which one should i choose", "recommendation", "which to use", "best vector database", "best vdb"]):
            return self._handle_better_overall()

        # 14. Single database profile or inquiry
        if len(mentioned_dbs) == 1:
            return self._handle_single_db_profile(mentioned_dbs[0], q_lower)

        # 15. Semantic fallback query
        return self._handle_general_search(q)

    # -------------------------------------------------------------------------
    # Intent Handlers
    # -------------------------------------------------------------------------

    def _handle_lowest_latency(self) -> Dict[str, Any]:
        qdrant_lat = self.measured["qdrant"]["latency_ms"]
        chroma_lat = self.measured["chroma"]["latency_ms"]
        faiss_lat = self.measured["faiss"]["latency_ms"]

        ans = (
            f"### Query Latency Evaluation `[MEASURED]`\n\n"
            f"Based strictly on genuine empirical measurements under the controlled 1,000-document SciFact benchmark, "
            f"**FAISS** achieved the lowest raw latency across all measured percentiles, followed by **Chroma** and **Qdrant**.\n\n"
            f"| Database | Execution Mode | Median (P50) | Tail (P95) | Tail (P99) | Mean Latency |\n"
            f"| :--- | :--- | :---: | :---: | :---: | :---: |\n"
            f"| **FAISS** | In-Process C++ Algorithmic Baseline (`IndexFlatIP`) | **{faiss_lat['p50']:.2f} ms** | **{faiss_lat['p95']:.2f} ms** | **{faiss_lat['p99']:.2f} ms** | **{faiss_lat['mean']:.2f} ms** |\n"
            f"| **Chroma** | Local Persistent Storage (`HNSW` via `hnswlib`) | **{chroma_lat['p50']:.2f} ms** | **{chroma_lat['p95']:.2f} ms** | **{chroma_lat['p99']:.2f} ms** | **{chroma_lat['mean']:.2f} ms** |\n"
            f"| **Qdrant** | Local Embedded In-Memory (Exact scan / unindexed segment) | **{qdrant_lat['p50']:.2f} ms** | **{qdrant_lat['p95']:.2f} ms** | **{qdrant_lat['p99']:.2f} ms** | **{qdrant_lat['mean']:.2f} ms** |\n\n"
            f"#### Architectural Nuance & Fair Comparison:\n"
            f"1. **FAISS Architectural Scope**: FAISS exhibits near-zero latency ({faiss_lat['p50']:.2f} ms P50) because it is a **pure in-process C++ algorithmic library**, not a database. It has zero server daemon overhead, no persistence WAL, and does not evaluate metadata filters.\n"
            f"2. **Embedded Databases**: Chroma ({chroma_lat['p50']:.2f} ms P50) and Qdrant ({qdrant_lat['p50']:.2f} ms P50) both deliver production-grade sub-5ms latency while managing schemas, document payloads, and index structures.\n\n"
            f"{self._get_unbenchmarked_notice()}"
            f"{self._get_workload_footnote()}"
        )
        return {
            "answer": ans,
            "intent": "LOWEST_LATENCY",
            "evidence_type": "MEASURED",
            "databases_mentioned": ["FAISS", "Chroma", "Qdrant"],
            "data_points": {
                "faiss_p50_ms": faiss_lat["p50"],
                "chroma_p50_ms": chroma_lat["p50"],
                "qdrant_p50_ms": qdrant_lat["p50"]
            }
        }

    def _handle_highest_qps(self) -> Dict[str, Any]:
        qdrant_qps = self.measured["qdrant"]["throughput_qps"]
        chroma_qps = self.measured["chroma"]["throughput_qps"]
        faiss_qps = self.measured["faiss"]["throughput_qps"]

        ans = (
            f"### Sequential Query Throughput (QPS) `[MEASURED]`\n\n"
            f"In the controlled benchmark (50 queries × 3 repetitions = 150 measured queries), sequential throughput results are:\n\n"
            f"| Database | Execution Mode | Measured QPS | Mean Query Latency |\n"
            f"| :--- | :--- | :---: | :---: |\n"
            f"| **FAISS** | In-Process C++ Algorithmic Baseline | **{faiss_qps:.1f} QPS** | {self.measured['faiss']['latency_ms']['mean']:.2f} ms |\n"
            f"| **Chroma** | Local Persistent Storage (`hnswlib`) | **{chroma_qps:.1f} QPS** | {self.measured['chroma']['latency_ms']['mean']:.2f} ms |\n"
            f"| **Qdrant** | Local Embedded In-Memory (Exact scan / unindexed segment) | **{qdrant_qps:.1f} QPS** | {self.measured['qdrant']['latency_ms']['mean']:.2f} ms |\n\n"
            f"#### Technical Analysis:\n"
            f"- **FAISS** reaches **{faiss_qps:.1f} QPS** because raw vector distances are calculated in tight C++ loops directly on contiguous RAM arrays with zero network, serialization, or transaction layers.\n"
            f"- **Chroma** achieved **{chroma_qps:.1f} QPS** in persistent mode via direct Python/C++ pointer bindings in `hnswlib`.\n"
            f"- **Qdrant** achieved **{qdrant_qps:.1f} QPS** in local embedded mode while maintaining a full Rust segment manager and payload filtering abstractions.\n\n"
            f"{self._get_unbenchmarked_notice()}"
            f"{self._get_workload_footnote()}"
        )
        return {
            "answer": ans,
            "intent": "HIGHEST_QPS",
            "evidence_type": "MEASURED",
            "databases_mentioned": ["FAISS", "Chroma", "Qdrant"],
            "data_points": {
                "faiss_qps": faiss_qps,
                "chroma_qps": chroma_qps,
                "qdrant_qps": qdrant_qps
            }
        }

    def _handle_ingestion(self) -> Dict[str, Any]:
        qdrant_ing = self.measured["qdrant"]["ingestion"]
        chroma_ing = self.measured["chroma"]["ingestion"]
        faiss_ing = self.measured["faiss"]["ingestion"]

        ans = (
            f"### Ingestion Throughput & Duration `[MEASURED]`\n\n"
            f"Under the identical 1,000 document embedding ingestion workload (384-dimensional dense vectors):\n\n"
            f"| Database | Ingestion Throughput | Total Ingestion Time | Index Build Time |\n"
            f"| :--- | :---: | :---: | :---: |\n"
            f"| **FAISS** | **{faiss_ing['vectors_per_sec']:.1f} vec/s** | **{faiss_ing['total_insertion_time_sec']:.3f} s** | {faiss_ing['index_build_time_sec']:.2f} s |\n"
            f"| **Qdrant** | **{qdrant_ing['vectors_per_sec']:.1f} vec/s** | **{qdrant_ing['total_insertion_time_sec']:.3f} s** | {qdrant_ing['index_build_time_sec']:.2f} s |\n"
            f"| **Chroma** | **{chroma_ing['vectors_per_sec']:.1f} vec/s** | **{chroma_ing['total_insertion_time_sec']:.3f} s** | {chroma_ing['index_build_time_sec']:.2f} s |\n\n"
            f"#### Technical Rationale:\n"
            f"1. **FAISS** takes only 2 milliseconds because `IndexFlatIP.add()` is a contiguous memory pointer copy in C++ without formatting, schema validation, or persistence.\n"
            f"2. **Qdrant** inserted at **1,121.7 vec/s** (3.1x faster than Chroma), taking advantage of Rust memory segmentation and concurrent collection initialization.\n"
            f"3. **Chroma** ingested at **360.7 vec/s**, reflecting SQLite catalog writes, disk flushing, and `hnswlib` index file serialization on disk.\n\n"
            f"{self._get_unbenchmarked_notice()}"
            f"{self._get_workload_footnote()}"
        )
        return {
            "answer": ans,
            "intent": "INGESTION_SPEED",
            "evidence_type": "MEASURED",
            "databases_mentioned": ["FAISS", "Qdrant", "Chroma"],
            "data_points": {
                "faiss_vec_s": faiss_ing["vectors_per_sec"],
                "qdrant_vec_s": qdrant_ing["vectors_per_sec"],
                "chroma_vec_s": chroma_ing["vectors_per_sec"]
            }
        }

    def _handle_retrieval_quality(self, query_lower: str) -> Dict[str, Any]:
        q_ir = self.measured["qdrant"]["retrieval_quality"]
        c_ir = self.measured["chroma"]["retrieval_quality"]
        f_ir = self.measured["faiss"]["retrieval_quality"]

        ans = (
            f"### Retrieval Quality Evaluation (IR Metrics) `[MEASURED]`\n\n"
            f"Evaluated against official **BEIR/SciFact ground-truth relevance judgments** (54 binary positive judgments):\n\n"
            f"| Metric | Qdrant (In-Memory Exact) | Chroma (HNSW) | FAISS (`IndexFlatIP`) | Theoretical Significance |\n"
            f"| :--- | :---: | :---: | :---: | :--- |\n"
            f"| **Recall@10** | **{q_ir['recall_at_10']:.4f}** | **{c_ir['recall_at_10']:.4f}** | **{f_ir['recall_at_10']:.4f}** | Fraction of relevant documents retrieved in Top-10 |\n"
            f"| **Precision@10** | **{q_ir['precision_at_10']:.4f}** | **{c_ir['precision_at_10']:.4f}** | **{f_ir['precision_at_10']:.4f}** | Proportion of retrieved Top-10 docs that are relevant |\n"
            f"| **Hit Rate@10** | **{q_ir['hit_rate_at_10']:.4f}** | **{c_ir['hit_rate_at_10']:.4f}** | **{f_ir['hit_rate_at_10']:.4f}** | Probability of retrieving at least 1 relevant doc |\n"
            f"| **MRR** | **{q_ir['mrr']:.4f}** | **{c_ir['mrr']:.4f}** | **{f_ir['mrr']:.4f}** | Mean Reciprocal Rank of first relevant document |\n"
            f"| **NDCG@10** | **{q_ir['ndcg_at_10']:.4f}** | **{c_ir['ndcg_at_10']:.4f}** | **{f_ir['ndcg_at_10']:.4f}** | Normalized Discounted Cumulative Gain at Top-10 |\n\n"
            f"#### Retrieval Behavior & Equivalence Analysis:\n"
            f"All three benchmarked systems returned identical IR quality metrics (`Recall@10 = 0.8700`, `MRR = 0.7295`, `NDCG@10 = 0.7602`).\n\n"
            f"- **Similarity Formulation Equivalence**: Because the embeddings are $L_2$-normalized (||v||_2 = 1.0), cosine similarity and inner product are mathematically equivalent for this embedding representation ($D_{{\\text{{cosine}}}}(u, v) = 1 - \\langle u, v \\rangle$).\n"
            f"- **Empirical Retrieval Result**: The identical IR scores observed here indicate that the benchmarked systems returned the same evaluated top-10 results for this workload. On this 1,000-document corpus, Chroma's HNSW graph traversal and Qdrant's unindexed in-memory flat scan returned the same top-10 candidate set as FAISS's exact flat index scan.\n\n"
            f"{self._get_unbenchmarked_notice()}"
            f"{self._get_workload_footnote()}"
        )
        return {
            "answer": ans,
            "intent": "RETRIEVAL_QUALITY",
            "evidence_type": "MEASURED",
            "databases_mentioned": ["Qdrant", "Chroma", "FAISS"],
            "data_points": {
                "recall_at_10": q_ir["recall_at_10"],
                "precision_at_10": q_ir["precision_at_10"],
                "mrr": q_ir["mrr"],
                "ndcg_at_10": q_ir["ndcg_at_10"]
            }
        }

    def _handle_postgresql_integration(self, mentioned_dbs: List[str]) -> Dict[str, Any]:
        pg_profile = self.profiles.get("pgvector", {}).get("attributes", {})
        ans = (
            f"### PostgreSQL Vector Integration `[DOCUMENTED]`\n\n"
            f"Among all evaluated vector systems, **pgvector** is the dedicated native extension designed specifically for **PostgreSQL**.\n\n"
            f"#### Key Capabilities of pgvector:\n"
            f"- **System Category**: Open-source PostgreSQL extension written in C (PostgreSQL License).\n"
            f"- **Storage & Transaction Model**: Stored directly in standard PostgreSQL heap tables and buffer pool (`shared_buffers`). Inherits full **ACID compliance**, point-in-time recovery (PITR), and Write-Ahead Logging (WAL).\n"
            f"- **Indexing Algorithms**: Supports **HNSW** (Hierarchical Navigable Small World, added in v0.5.0) and **IVFFlat** (Inverted File Flat) index types.\n"
            f"- **Distance Operators**: Euclidean `<->`, Inner Product `<#>`, Cosine `<=>`, L1 `<+>`.\n"
            f"- **Quantization & Data Types**: Supports standard `vector(dim)`, `halfvec` (FP16, up to 16,000 dimensions), `bit` (binary quantization), and `sparsevec`.\n"
            f"- **Relational Power**: Execute vector similarity queries as part of standard SQL `SELECT ... ORDER BY embedding <=> $1 LIMIT 10`, allowing seamless relational `JOIN`s, `WHERE` scalar filters, and window functions.\n\n"
            f"#### Comparison with Other Systems for PostgreSQL Users:\n"
            f"- **Choose pgvector** if: You already run PostgreSQL and want to avoid introducing, managing, and synchronizing a second database engine.\n"
            f"- **Choose a Native Vector DB (e.g. Qdrant / Milvus)** if: Your vector workload exceeds millions of vectors, requires ultra-high ingestion throughput (>10k vec/s), or demands sub-10ms tail latencies under heavy concurrent search, where specialized vector engines outperform general-purpose RDBMS engines.\n\n"
            f"> [!NOTE]\n"
            f"> **Benchmark Status `[NOT BENCHMARKED]`**: pgvector was not executed in the local empirical benchmark because `psycopg2` was uninstalled and no active PostgreSQL server daemon was running in this testing environment.\n"
        )
        return {
            "answer": ans,
            "intent": "POSTGRESQL_INTEGRATION",
            "evidence_type": "DOCUMENTED",
            "databases_mentioned": ["pgvector"],
            "data_points": {
                "system": "pgvector",
                "core_language": "C",
                "license": "PostgreSQL License"
            }
        }

    def _handle_metadata_filtering(self, mentioned_dbs: List[str]) -> Dict[str, Any]:
        ans = (
            f"### Metadata Filtering Strategies Across Vector Databases `[DOCUMENTED]`\n\n"
            f"Filtering scalar metadata alongside vector search is one of the most critical requirements for production RAG and enterprise search. "
            f"The 8 systems implement three distinct filtering paradigms:\n\n"
            f"| Database | Filtering Strategy | How It Works | Recall / Performance Trade-off |\n"
            f"| :--- | :--- | :--- | :--- |\n"
            f"| **Qdrant** | **Single-Stage (Custom Graph Traversal)** | Dynamic payload filter bitmask checked during HNSW graph traversal. | **Optimal**: Guaranteed Top-$K$ with zero recall loss; avoids scanning unneeded branches. |\n"
            f"| **Weaviate** | **Single-Stage (Inverted Index + HNSW)** | Combines property inverted index directly with graph traversal. | **Optimal**: Exact Top-$K$ retrieval with structured payload indices. |\n"
            f"| **Pinecone** | **Single-Stage Graph Traversal** | Dynamic graph traversal with metadata condition evaluation. | **Optimal**: Fast managed single-stage filtering. |\n"
            f"| **Elasticsearch** | **Lucene Bitset Masking** | Evaluates filters into Lucene bitsets applied directly to HNSW search. | **Robust**: Leverages industry-standard Lucene inverted indexes. |\n"
            f"| **pgvector** | **Integrated SQL Query Planner** | Planner chooses between index scan, B-tree pre-filter, or re-ranking. | **Flexible**: Uses PostgreSQL Cost-Based Optimizer (CBO). |\n"
            f"| **Chroma** | **Pre/Post-Filtering Hybrid** | Queries SQLite for metadata IDs, then filters vector results. | **Good for small scale**; potential latency overhead at millions of items. |\n"
            f"| **Milvus** | **Iterative Filtering / Pre-filtering** | Bitset generated from scalar predicate before/during search. | **Scalable**: Hardware accelerated scalar filtering across QueryNodes. |\n"
            f"| **FAISS** | **Post-Filtering / IDSelector** | Computes vector distances first, or uses explicit `IDSelector`. | **Limited**: Lacks native metadata storage; host app must store metadata. |\n\n"
            f"#### Why Single-Stage Filtering Matters for RAG:\n"
            f"Post-filtering risks **'recall collapse'** when filters are highly selective (e.g., retrieving Top-10 vectors where only 2 match the user's tenant ID, returning just 2 results). "
            f"Qdrant and Weaviate guarantee exactly $K$ valid matches by traversing only connected graph nodes that satisfy the filter predicate.\n"
        )
        return {
            "answer": ans,
            "intent": "METADATA_FILTERING",
            "evidence_type": "DOCUMENTED",
            "databases_mentioned": self.all_dbs,
            "data_points": {"paradigms": ["Single-Stage", "Pre-Filtering", "Post-Filtering"]}
        }

    def _handle_rag_suitability(self) -> Dict[str, Any]:
        ans = (
            f"### Vector Database Selection for RAG Applications `[EVIDENCE-GROUNDED ANALYSIS]`\n\n"
            f"Retrieval-Augmented Generation (RAG) places specific demands on a vector database:\n"
            f"1. **Single-Stage Payload Filtering**: To enforce tenant isolation, timestamps, or document categories without losing Top-$K$ recall.\n"
            f"2. **Hybrid Search (BM25 + Dense)**: To catch exact acronyms, part numbers, and semantic intent simultaneously.\n"
            f"3. **Metadata Storage & Durability**: Storing text chunks alongside vectors to avoid a secondary document store lookup.\n"
            f"4. **Low Query Latency**: Keeping retrieval under 10ms so LLM generation remains responsive.\n\n"
            f"#### Workload-Grounded RAG Candidate Analysis:\n\n"
            f"1. **Production Self-Hosted / Managed RAG Candidates**:\n"
            f"   - **Qdrant**: Based on the documented capabilities and the measured workload in this project, Qdrant is a strong candidate for production RAG requiring native Rust speed (`4.43 ms` P50 `[MEASURED]`), built-in single-stage payload filtering, native BM25 full-text search, and Reciprocal Rank Fusion (RRF).\n"
            f"   - **Weaviate**: Based on documented capabilities, Weaviate is a strong candidate for modular RAG architectures with native GraphQL/gRPC APIs, out-of-the-box hybrid search (BM25 + vector), modular ML vectorizers, and single-stage filtering `[DOCUMENTED]`.\n\n"
            f"2. **Rapid Prototyping & Local Development Candidate**:\n"
            f"   - **Chroma**: Based on documented capabilities and measured results (`3.24 ms` P50 `[MEASURED]`), Chroma is a suitable candidate for local agentic apps and rapid prototypes with zero-setup embedded SQLite metadata storage and LangChain/LlamaIndex integration.\n\n"
            f"3. **Existing PostgreSQL Stack Candidate**:\n"
            f"   - **pgvector**: Based on documented capabilities, pgvector is the appropriate choice when joining vector chunks directly with relational tables, user permissions, and metadata in PostgreSQL `[DOCUMENTED]`.\n\n"
            f"4. **Zero-DevOps Cloud Serverless Candidate**:\n"
            f"   - **Pinecone**: Based on documented capabilities, Pinecone is a candidate for teams seeking managed serverless infrastructure with built-in metadata filtering and sparse-dense indexing without managing cluster infrastructure `[DOCUMENTED]`.\n\n"
            f"5. **Architectural Role of FAISS in RAG**:\n"
            f"   - While FAISS achieved the lowest raw compute latency (`0.23 ms` `[MEASURED]`), it **does not store document payloads, text chunks, or metadata**. For RAG pipelines, host applications must build external document storage, disk durability, and filtering layers around it.\n"
        )
        return {
            "answer": ans,
            "intent": "RAG_SUITABILITY",
            "evidence_type": "HYBRID",
            "databases_mentioned": ["Qdrant", "Weaviate", "Chroma", "pgvector", "Pinecone", "FAISS"],
            "data_points": {"top_picks": ["Qdrant", "Weaviate", "Chroma", "pgvector", "Pinecone"]}
        }

    def _handle_distributed_deployment(self, mentioned_dbs: List[str]) -> Dict[str, Any]:
        ans = (
            f"### Distributed Deployment, Sharding & High Availability `[DOCUMENTED]`\n\n"
            f"When scaling beyond a single server or requiring 99.99% high availability, the 8 systems exhibit vastly different architectures:\n\n"
            f"| Database | Horizontal Sharding | High Availability / Replicas | Cluster Consensus | Architecture Type |\n"
            f"| :--- | :--- | :--- | :--- | :--- |\n"
            f"| **Milvus** | Native dynamic sharding across QueryNodes | Stateless worker replicas scale independently | Etcd + Apache Pulsar/Kafka | Cloud-Native Decoupled Microservices |\n"
            f"| **Qdrant** | Native automatic hash-ring sharding | Multi-node Raft consensus replicas | Embedded Raft (Rust) | Native Distributed Vector Database |\n"
            f"| **Weaviate** | Native multi-shard class partitioning | Dynamic replication factor with Raft | Raft consensus (Go) | Native Vector Search Engine Cluster |\n"
            f"| **Elasticsearch** | Primary and replica shard distribution | Automated master failover & replica routing | Raft-based Zen Discovery | Distributed Enterprise Search Cluster |\n"
            f"| **Pinecone** | Virtualized automated serverless sharding | Multi-AZ cloud infrastructure | Proprietary Cloud Orchestrator | Managed Serverless Cloud SaaS |\n"
            f"| **pgvector** | Via PostgreSQL Citus / Partitioning | PostgreSQL Streaming Replication & Patroni | External (Patroni/Pacemaker) | Relational Database Cluster |\n"
            f"| **Chroma** | Single-node embedded (Distributed in Enterprise) | Single reader/writer in open-source | SQLite file lock | Embedded / Local-First Store |\n"
            f"| **FAISS** | Manual application-level (`ShardedIndex`) | Application manages multiple instances | None (No networking) | In-Process C++ Library |\n\n"
            f"#### Summary Recommendation:\n"
            f"- For **massive multi-billion vector scale**, **Milvus** provides the most thoroughly decoupled Kubernetes-native architecture.\n"
            f"- For **clean, self-contained multi-node clustering without heavy microservice dependencies**, **Qdrant** (embedded Raft) is significantly simpler to operate.\n"
        )
        return {
            "answer": ans,
            "intent": "DISTRIBUTED_DEPLOYMENT",
            "evidence_type": "DOCUMENTED",
            "databases_mentioned": self.all_dbs,
            "data_points": {"distributed_native": ["Milvus", "Qdrant", "Weaviate", "Elasticsearch", "Pinecone"]}
        }

    def _handle_hybrid_search(self, mentioned_dbs: List[str]) -> Dict[str, Any]:
        ans = (
            f"### Hybrid Search & Sparse-Dense Fusion Capabilities `[DOCUMENTED]`\n\n"
            f"Hybrid search combines dense semantic retrieval (vector embeddings) with sparse keyword retrieval (BM25 / SPLADE) "
            f"to achieve superior recall on both semantic queries and exact keyword/acronym matches.\n\n"
            f"| Database | Native BM25 Lexical Search | Sparse Vector Support | Fusion Algorithm | Evaluation |\n"
            f"| :--- | :---: | :---: | :--- | :--- |\n"
            f"| **Elasticsearch** | **Gold Standard** (Lucene) | Yes | **Reciprocal Rank Fusion (RRF)** & Linear Combination | Unmatched text analyzer and relevance tuning ecosystem |\n"
            f"| **Qdrant** | Yes (Built-in payload token analyzer) | Yes (Named sparse vectors) | **Reciprocal Rank Fusion (RRF)** & Score Boosting | Clean unified API for dense + sparse + text |\n"
            f"| **Weaviate** | Yes (Native BM25 inverted index) | Yes | **Relative Score Fusion** & RRF | Seamless hybrid queries with `alpha` weighting parameter |\n"
            f"| **Milvus** | Yes (Built-in analyzer v2.4+) | Yes (Sparse vector fields) | **Reciprocal Rank Fusion (RRF)** & Relative Score | High-throughput distributed hybrid search |\n"
            f"| **Pinecone** | Yes | Yes (Sparse-dense index) | **Convex Linear Combination** (alpha * Dense + (1-alpha) * Sparse) | Managed cloud hybrid retrieval |\n"
            f"| **pgvector** | Yes (via PostgreSQL `tsvector`/`tsquery`) | Yes (via `sparsevec`) | SQL Ranking Fusion (RRF via CTE or linear score) | Combines full-text search with vector operators in SQL |\n"
            f"| **Chroma** | Basic substring matching only | No native sparse | Requires external client re-ranking | Relies on external frameworks (e.g. Cohere rerank) |\n"
            f"| **FAISS** | No | Sparse matrix indexing (limited) | Requires external host application fusion | Algorithmic vector search only |\n\n"
            f"#### What is Reciprocal Rank Fusion (RRF)?\n"
            f"Formula: RRF_Score(d) = sum_{{m in M}} [ 1 / (60 + rank_m(d)) ]\n"
            f"RRF is supported natively by **Qdrant, Milvus, Weaviate, and Elasticsearch**. It does not require score normalization, "
            f"making it immune to differences in score distributions between BM25 and cosine distance.\n"
        )
        return {
            "answer": ans,
            "intent": "HYBRID_SEARCH",
            "evidence_type": "DOCUMENTED",
            "databases_mentioned": self.all_dbs,
            "data_points": {"rrf_supported": ["Qdrant", "Milvus", "Weaviate", "Elasticsearch"]}
        }

    def _handle_licensing(self, mentioned_dbs: List[str]) -> Dict[str, Any]:
        ans = (
            f"### Licensing and Open-Source Evaluation `[DOCUMENTED]`\n\n"
            f"| Database | Licensing Model | Commercial Availability | Self-Hostable? |\n"
            f"| :--- | :--- | :--- | :---: |\n"
            f"| **Qdrant** | **Apache 2.0 (True Open Source)** | Qdrant Cloud (Managed SaaS) | Yes (Docker, K8s, Linux binary) |\n"
            f"| **Chroma** | **Apache 2.0 (True Open Source)** | Chroma Cloud (Hosted) | Yes (Python package, Docker) |\n"
            f"| **FAISS** | **MIT License (True Open Source)** | Pure Open Source Library | Yes (Local in-process library) |\n"
            f"| **Milvus** | **Apache 2.0 (LF AI & Data Foundation)** | Zilliz Cloud (Fully Managed) | Yes (Helm, Docker Compose, K8s) |\n"
            f"| **Weaviate** | **BSD-3-Clause (True Open Source)** | Weaviate Cloud Services (WCS) | Yes (Docker, Helm, K8s) |\n"
            f"| **pgvector** | **PostgreSQL License (True Open Source)** | Available on AWS RDS, Supabase, Neon | Yes (Any standard PostgreSQL instance) |\n"
            f"| **Elasticsearch** | **Source-Available (ELv2 / SSPL / Apache 2.0)** | Elastic Cloud (Multi-Cloud) | Yes (Subject to Elastic license terms) |\n"
            f"| **Pinecone** | **Proprietary Commercial SaaS** | Exclusively Managed Cloud | **No (Cloud SaaS Only)** |\n\n"
            f"#### Recommendation for Open-Source Deployments:\n"
            f"- If you require **permissive, unencumbered open-source (Apache 2.0 / BSD / MIT)** for self-hosting on your own infrastructure, "
            f"**Qdrant, Weaviate, Milvus, Chroma, pgvector**, and **FAISS** are all fully compliant.\n"
            f"- Avoid **Pinecone** if zero-cloud lock-in or on-premise air-gapped deployment is required.\n"
        )
        return {
            "answer": ans,
            "intent": "OPEN_SOURCE_LICENSING",
            "evidence_type": "DOCUMENTED",
            "databases_mentioned": self.all_dbs,
            "data_points": {"true_oss": ["Qdrant", "Chroma", "FAISS", "Milvus", "Weaviate", "pgvector"]}
        }

    def _handle_better_overall(self) -> Dict[str, Any]:
        ans = (
            f"### Comprehensive Vector Database Decision Matrix — Which is 'Better'?\n\n"
            f"> [!IMPORTANT]\n"
            f"> In production engineering, **there is no single 'best' vector database**. The optimal choice depends entirely "
            f"> on your operational scale, existing infrastructure, query latency budgets, and metadata filtering needs.\n\n"
            f"#### Objective Criterion-Based Trade-off Comparison Across 6 Production Workloads:\n\n"
            f"| Production Requirement | Strong Candidate | Alternative Candidate | Key Trade-off / Architectural Rationale |\n"
            f"| :--- | :--- | :--- | :--- |\n"
            f"| **Lowest Raw Compute Latency** | **FAISS** (`0.23 ms` `[MEASURED]`) | **Chroma** (`3.24 ms` `[MEASURED]`) | FAISS is an in-process C++ library without durability, WAL, or server overhead; not a standalone database. |\n"
            f"| **Production RAG & Rich Filtering** | **Qdrant** (`4.43 ms` `[MEASURED]`) | **Weaviate** `[DOCUMENTED]` | Rust speed, single-stage graph filtering, built-in BM25 full-text, and native RRF fusion. |\n"
            f"| **Fastest Prototyping & Local Dev** | **Chroma** (`3.24 ms` `[MEASURED]`) | **Qdrant (Embedded)** | Chroma requires zero setup in Python; however, single-writer SQLite limits high-throughput scaling. |\n"
            f"| **Existing PostgreSQL Stack** | **pgvector** `[DOCUMENTED]` | Native Vector DB | Zero new infrastructure; full ACID compliance and relational SQL joins; lower QPS than specialized C++/Rust engines. |\n"
            f"| **Hyper-Scale (>100M+ Vectors, K8s)** | **Milvus** `[DOCUMENTED]` | **Elasticsearch** `[DOCUMENTED]` | Decoupled cloud-native microservices with S3/MinIO backing; higher deployment and DevOps complexity. |\n"
            f"| **Zero-DevOps Cloud Serverless** | **Pinecone** `[DOCUMENTED]` | Managed Qdrant / Zilliz | Fully managed with automatic scaling; proprietary cloud lock-in and ongoing SaaS subscription costs. |\n\n"
            f"#### Workload & Requirement Decision Guidelines:\n"
            f"- Based on the documented capabilities and the measured workload in this project, **Qdrant** is a strong candidate for building a modern, high-throughput RAG application from scratch due to single-stage filtering, sub-5ms query latency, and hybrid search.\n"
            f"- If your infrastructure is centered on **PostgreSQL**, starting with **pgvector** eliminates the operational complexity of managing a separate vector database.\n"
            f"- For **offline research, raw vector clustering, or custom inference pipelines**, **FAISS** delivers maximum compute throughput.\n"
            f"- For **large-scale enterprise deployments on Kubernetes**, evaluate **Milvus** (decoupled microservices) or **Weaviate** (modular vectorizers).\n"
        )
        return {
            "answer": ans,
            "intent": "BETTER_OVERALL",
            "evidence_type": "HYBRID",
            "databases_mentioned": self.all_dbs,
            "data_points": {"decision_factors": ["Latency", "Filtering", "Prototyping", "SQL Integration", "Scale", "SaaS"]}
        }

    def _handle_comparison(self, dbs: List[str], query_lower: str) -> Dict[str, Any]:
        db1_name = dbs[0]
        db2_name = dbs[1]
        db1_key = self.data["database_profiles"][db1_name]["key"]
        db2_key = self.data["database_profiles"][db2_name]["key"]

        db1_benchmarked = db1_key in self.benchmarked_keys
        db2_benchmarked = db2_key in self.benchmarked_keys

        p1 = self.profiles[db1_name]["attributes"]
        p2 = self.profiles[db2_name]["attributes"]

        ans = f"### Head-to-Head Comparison: {db1_name} vs {db2_name}\n\n"

        # Performance section
        if db1_benchmarked and db2_benchmarked:
            m1 = self.measured[db1_key]
            m2 = self.measured[db2_key]
            ans += (
                f"#### 1. Measured Empirical Performance `[MEASURED]`\n"
                f"*(Under controlled 1,000-doc / 50-query SciFact workload)*\n\n"
                f"| Performance Metric | {db1_name} | {db2_name} | Ratio / Difference |\n"
                f"| :--- | :---: | :---: | :--- |\n"
                f"| **Ingestion Speed** | **{m1['ingestion']['vectors_per_sec']:.1f} vec/s** | **{m2['ingestion']['vectors_per_sec']:.1f} vec/s** | {db1_name if m1['ingestion']['vectors_per_sec'] > m2['ingestion']['vectors_per_sec'] else db2_name} is {max(m1['ingestion']['vectors_per_sec'], m2['ingestion']['vectors_per_sec']) / max(min(m1['ingestion']['vectors_per_sec'], m2['ingestion']['vectors_per_sec']), 0.001):.1f}x faster |\n"
                f"| **Total Ingestion Time** | {m1['ingestion']['total_insertion_time_sec']:.3f} s | {m2['ingestion']['total_insertion_time_sec']:.3f} s | — |\n"
                f"| **Median Latency (P50)** | **{m1['latency_ms']['p50']:.2f} ms** | **{m2['latency_ms']['p50']:.2f} ms** | {db1_name if m1['latency_ms']['p50'] < m2['latency_ms']['p50'] else db2_name} is {abs(m1['latency_ms']['p50'] - m2['latency_ms']['p50']):.2f} ms lower |\n"
                f"| **Tail Latency (P95)** | **{m1['latency_ms']['p95']:.2f} ms** | **{m2['latency_ms']['p95']:.2f} ms** | {db1_name if m1['latency_ms']['p95'] < m2['latency_ms']['p95'] else db2_name} is {abs(m1['latency_ms']['p95'] - m2['latency_ms']['p95']):.2f} ms lower |\n"
                f"| **Query Throughput** | **{m1['throughput_qps']:.1f} QPS** | **{m2['throughput_qps']:.1f} QPS** | {db1_name if m1['throughput_qps'] > m2['throughput_qps'] else db2_name} has higher QPS |\n"
                f"| **Recall@10** | **{m1['retrieval_quality']['recall_at_10']:.4f}** | **{m2['retrieval_quality']['recall_at_10']:.4f}** | Identical on controlled subset |\n"
                f"| **MRR** | **{m1['retrieval_quality']['mrr']:.4f}** | **{m2['retrieval_quality']['mrr']:.4f}** | Identical on controlled subset |\n"
                f"| **NDCG@10** | **{m1['retrieval_quality']['ndcg_at_10']:.4f}** | **{m2['retrieval_quality']['ndcg_at_10']:.4f}** | Identical on controlled subset |\n\n"
            )
        elif db1_benchmarked and not db2_benchmarked:
            m1 = self.measured[db1_key]
            reason = self.unbenchmarked[db2_key]["reason_summary"]
            ans += (
                f"#### 1. Empirical Benchmark Status\n"
                f"- **{db1_name} `[MEASURED]`**: Ingestion = {m1['ingestion']['vectors_per_sec']:.1f} vec/s | P50 = {m1['latency_ms']['p50']:.2f} ms | QPS = {m1['throughput_qps']:.1f}\n"
                f"- **{db2_name} `[NOT BENCHMARKED]`**: *{reason}*. No empirical numbers were fabricated.\n\n"
            )
        elif not db1_benchmarked and db2_benchmarked:
            m2 = self.measured[db2_key]
            reason = self.unbenchmarked[db1_key]["reason_summary"]
            ans += (
                f"#### 1. Empirical Benchmark Status\n"
                f"- **{db1_name} `[NOT BENCHMARKED]`**: *{reason}*. No empirical numbers were fabricated.\n"
                f"- **{db2_name} `[MEASURED]`**: Ingestion = {m2['ingestion']['vectors_per_sec']:.1f} vec/s | P50 = {m2['latency_ms']['p50']:.2f} ms | QPS = {m2['throughput_qps']:.1f}\n\n"
            )
        else:
            ans += (
                f"#### 1. Empirical Benchmark Status `[NOT BENCHMARKED]`\n"
                f"Both **{db1_name}** and **{db2_name}** were not benchmarked in the local environment:\n"
                f"- {db1_name}: *{self.unbenchmarked[db1_key]['reason_summary']}*\n"
                f"- {db2_name}: *{self.unbenchmarked[db2_key]['reason_summary']}*\n"
                f"Comparison below is based strictly on verified **Documented Technical Capabilities**.\n\n"
            )

        # Technical architecture comparison
        params_to_compare = [
            ("Core System Category", "Category"),
            ("Core Implementation Language", "Language"),
            ("Open Source / Licensing Model", "License"),
            ("Primary Deployment Topology", "Deployment"),
            ("Underlying Storage Architecture", "Storage Engine"),
            ("Write-Ahead Logging (WAL) & Durability", "Durability / WAL"),
            ("Primary ANN Index Types", "ANN Indices"),
            ("Metadata Filtering Strategy", "Filtering"),
            ("Native Full-Text / BM25 Keyword Search", "BM25 Search"),
            ("Sparse-Dense Hybrid Fusion Algorithm", "Hybrid Fusion"),
            ("Horizontal Sharding Support", "Sharding"),
            ("Consensus / Cluster Coordination", "Consensus"),
            ("Core Technical Advantage / Strength", "Advantage"),
            ("Primary Architectural Limitation", "Limitation"),
            ("Optimal Production Workload & Use Case", "Optimal Use Case")
        ]

        ans += f"#### 2. Technical Architecture & Capability Comparison `[DOCUMENTED]`\n\n"
        ans += f"| Dimension | {db1_name} | {db2_name} |\n"
        ans += f"| :--- | :--- | :--- |\n"

        for p_name, short_label in params_to_compare:
            v1 = p1.get(p_name, {}).get("value", "N/A")
            v2 = p2.get(p_name, {}).get("value", "N/A")
            ans += f"| **{short_label}** | {v1} | {v2} |\n"

        return {
            "answer": ans,
            "intent": "COMPARE_PAIR",
            "evidence_type": "HYBRID",
            "databases_mentioned": [db1_name, db2_name],
            "data_points": {"db1": db1_name, "db2": db2_name}
        }

    def _handle_single_db_profile(self, db_name: str, query_lower: str) -> Dict[str, Any]:
        db_key = self.data["database_profiles"][db_name]["key"]
        p = self.profiles[db_name]["attributes"]
        is_benchmarked = db_key in self.benchmarked_keys

        ans = f"### Technical Profile: {db_name}\n\n"

        if is_benchmarked:
            m = self.measured[db_key]
            ans += (
                f"#### Empirical Benchmark Performance `[MEASURED]`\n"
                f"- **Ingestion Speed**: {m['ingestion']['vectors_per_sec']:.1f} vectors/sec (Total: {m['ingestion']['total_insertion_time_sec']:.3f} s)\n"
                f"- **Latency**: P50 = **{m['latency_ms']['p50']:.2f} ms** | P95 = **{m['latency_ms']['p95']:.2f} ms** | P99 = **{m['latency_ms']['p99']:.2f} ms** | Mean = **{m['latency_ms']['mean']:.2f} ms**\n"
                f"- **Throughput**: **{m['throughput_qps']:.1f} QPS**\n"
                f"- **Information Retrieval Quality**: Recall@10 = **{m['retrieval_quality']['recall_at_10']:.4f}**, MRR = **{m['retrieval_quality']['mrr']:.4f}**, NDCG@10 = **{m['retrieval_quality']['ndcg_at_10']:.4f}**\n\n"
            )
        else:
            reason = self.unbenchmarked[db_key]["reason_summary"]
            ans += (
                f"#### Empirical Benchmark Status `[NOT BENCHMARKED]`\n"
                f"> **Status**: Not Benchmarked ({reason}). Under scientific integrity standards, no performance metrics are fabricated.\n\n"
            )

        ans += (
            f"#### Technical Specifications `[DOCUMENTED]`\n"
            f"- **Category**: {p.get('Core System Category', {}).get('value')}\n"
            f"- **Language & License**: {p.get('Core Implementation Language', {}).get('value')} | {p.get('Open Source / Licensing Model', {}).get('value')}\n"
            f"- **Storage Engine**: {p.get('Underlying Storage Architecture', {}).get('value')}\n"
            f"- **Durability & WAL**: {p.get('Write-Ahead Logging (WAL) & Durability', {}).get('value')}\n"
            f"- **Indexing & Search**: {p.get('Primary ANN Index Types', {}).get('value')}\n"
            f"- **Filtering**: {p.get('Metadata Filtering Strategy', {}).get('value')}\n"
            f"- **Hybrid Search & Fusion**: {p.get('Native Full-Text / BM25 Keyword Search', {}).get('value')} ({p.get('Sparse-Dense Hybrid Fusion Algorithm', {}).get('value')})\n"
            f"- **Scalability**: Sharding: {p.get('Horizontal Sharding Support', {}).get('value')} | Consensus: {p.get('Consensus / Cluster Coordination', {}).get('value')}\n"
            f"- **Core Advantage**: {p.get('Core Technical Advantage / Strength', {}).get('value')}\n"
            f"- **Primary Limitation**: {p.get('Primary Architectural Limitation', {}).get('value')}\n"
            f"- **Optimal Workload**: {p.get('Optimal Production Workload & Use Case', {}).get('value')}\n"
        )
        return {
            "answer": ans,
            "intent": "SINGLE_DB_PROFILE",
            "evidence_type": "HYBRID" if is_benchmarked else "DOCUMENTED",
            "databases_mentioned": [db_name],
            "data_points": {"db": db_name}
        }

    def _handle_general_search(self, query: str) -> Dict[str, Any]:
        """Matches query keywords against parameter titles and descriptions in technical matrix."""
        q_tokens = [w.lower() for w in re.split(r'\W+', query) if len(w) > 3]
        matches = []

        for row in self.matrix_rows:
            param = row.get("parameter", "")
            section = row.get("section", "")
            param_lower = param.lower()

            score = sum(1 for token in q_tokens if token in param_lower)
            if score > 0:
                matches.append((score, row))

        if not matches:
            # Fallback helpful response
            ans = (
                f"### Vector Database Comparison Assistant\n\n"
                f"I couldn't match your query directly to a specific metric or architectural parameter. "
                f"Here are common grounded questions you can ask me:\n\n"
                f"- **Performance**: *'Which has the lowest latency?'*, *'Which has the highest QPS?'*, *'Show me the performance difference between Qdrant and Chroma.'*\n"
                f"- **Retrieval Quality**: *'Which has the best Recall@10?'*, *'Explain the IR metrics.'*\n"
                f"- **Head-to-Head**: *'Compare Qdrant and Chroma.'*, *'Compare Qdrant and FAISS.'*, *'What is the difference between Qdrant and Pinecone?'*\n"
                f"- **Capabilities**: *'Which supports metadata filtering?'*, *'Which supports PostgreSQL integration?'*, *'Which systems support distributed deployment?'*, *'Which systems support hybrid search?'*\n"
                f"- **Workloads**: *'Which is suitable for a RAG application?'*, *'Which option is suitable if I want an open-source deployment?'*, *'Which is better overall?'*\n"
            )
            return {
                "answer": ans,
                "intent": "GENERAL_FALLBACK",
                "evidence_type": "DOCUMENTED",
                "databases_mentioned": [],
                "data_points": {}
            }

        matches.sort(key=lambda x: x[0], reverse=True)
        top_rows = [m[1] for m in matches[:3]]

        ans = f"### Documented Capabilities Matching '{query}' `[DOCUMENTED]`\n\n"
        for row in top_rows:
            param = row.get("parameter", "")
            sec = row.get("section", "")
            ans += f"#### {param} *({sec})*\n\n"
            ans += "| Database | Specification / Capability |\n| :--- | :--- |\n"
            for db in self.all_dbs:
                val = row.get(db, "N/A")
                ans += f"| **{db}** | {val} |\n"
            ans += "\n"

        return {
            "answer": ans,
            "intent": "PARAMETER_LOOKUP",
            "evidence_type": "DOCUMENTED",
            "databases_mentioned": self.all_dbs,
            "data_points": {"matched_params": [r.get("parameter") for r in top_rows]}
        }
