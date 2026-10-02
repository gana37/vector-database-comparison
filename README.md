# Vector Database Comparative Analysis and Benchmarking Platform

An end-to-end engineering platform and empirical benchmarking system evaluating **8 Vector Databases and Search Engines** using common corpus data, embedding models, queries, and Information Retrieval (IR) evaluation metrics.

Built for the technical assessment requirement at **AgentAnalytics.AI**:
> *"1 Excel sheet, comparison analysis on 8 VDBs. Keep it exhaustive, technical depth."*

**Primary Deliverable**: [`outputs/excel/VDB_Comprehensive_Analysis.xlsx`](file:///c:/Users/User/Downloads/VED-Task/outputs/excel/VDB_Comprehensive_Analysis.xlsx) *(Strictly 1 single comprehensive worksheet: `VDB_Master_Comparison`)*

---

## Executive Summary & Scope Distinction

This platform provides both an exhaustive architectural evaluation across all 8 vector database systems and empirical benchmark measurements under controlled experimental conditions.

### Scope Distinction
1. **Technical Comparison (All 8 Systems)**:
   - **Qdrant**, **Chroma**, **FAISS**, **Milvus**, **Weaviate**, **pgvector**, **Elasticsearch**, and **Pinecone**.
   - Factual, multi-dimensional analysis spanning underlying storage engines, indexing algorithms (HNSW, IVF, PQ/SQ/BQ), filtering paradigms (single-stage graph traversal vs pre/post-filtering), hybrid sparse-dense fusion, distributed sharding, high availability, security, and operational TCO.
2. **Empirically Benchmarked Systems (3 Systems)**:
   - **Qdrant** (Local Embedded In-Memory Mode)
   - **Chroma** (Local Persistent Storage Mode)
   - **FAISS** (In-Process Algorithmic Baseline, Flat Index)
3. **Systems Not Benchmarked (5 Systems)**:
   - **Milvus**: Not benchmarked (*`pymilvus` client driver not installed; requires external cluster/daemon*).
   - **Weaviate**: Not benchmarked (*`weaviate-client` driver not installed; requires external cluster/daemon*).
   - **pgvector**: Not benchmarked (*`psycopg2` driver not installed; requires active PostgreSQL server daemon with pgvector extension*).
   - **Elasticsearch**: Not benchmarked (*`elasticsearch` client driver not installed; requires active Elasticsearch cluster daemon*).
   - **Pinecone**: Not benchmarked (*Cloud SaaS; `PINECONE_API_KEY` environment variable not configured to prevent unauthorized API requests and avoid unintended cloud billing*).

> [!IMPORTANT]
> **Strict Scientific Integrity**: In strict compliance with experimental standards, **no benchmark numbers are fabricated, estimated, or simulated** for unbenchmarked systems. Systems that were not executed in the live benchmark harness are clearly identified with their exact technical root cause.

> [!WARNING]
> **Controlled Workload Scope & Non-Generalization Notice**:
> The benchmark results presented here apply **only to this controlled workload** (1,000 documents, 50 queries, 384 dimensions, Top-$K=10$, local execution) and **should not be treated as universal performance claims**. Vector search performance in production depends heavily on vector dimensionality, index quantization, dataset scale (millions/billions of vectors), filtering selectivity, concurrency, network topology, and server hardware.

> [!NOTE]
> **FAISS Architectural Scope**:
> **FAISS is an in-process vector similarity-search library and algorithmic baseline, not a full client-server vector database**. It does not include a network server daemon, client-server API protocol, disk durability / Write-Ahead Log (WAL), dynamic CRUD document mutations, metadata payload storage and filtering, authentication/RBAC, or distributed clustering. In this platform, FAISS serves as a compute baseline representing raw hardware throughput and distance calculation efficiency rather than a direct database peer.

---

## Controlled Benchmark Workload Specification

The empirical benchmark was executed under strictly identical experimental parameters across all evaluated engines:

| Parameter | Specification | Purpose / Notes |
| :--- | :--- | :--- |
| **Corpus Dataset** | **1,000 documents** | Deterministically subsampled (`seed=42`) from `BEIR/SciFact` (TU Darmstadt mirror). |
| **Query Set** | **50 queries** | Matched deterministic queries from SciFact test/train splits. |
| **Ground-Truth Judgments**| **54 relevance judgments** | Binary positive relevance judgments mapped directly within the 1,000-doc corpus subset. |
| **Embedding Model** | `all-MiniLM-L6-v2` | Dense sentence transformer executed via ONNX Runtime (`v1.30.0`) with HuggingFace Tokenizers. |
| **Embedding Dimension** | **384-dimensional** | Generated once, validated for non-null/finite float32 values, and saved locally as numpy arrays. |
| **Vector Normalization** | **$L_2$ Unit Normalization** | $\|v\|_2 = 1.0$, ensuring Cosine Similarity equals Inner Product (Dot Product). |
| **Top-K Retrieval** | **Top-$K = 10$** | 10 nearest neighbors retrieved per query. |
| **Repetitions** | **3 repetitions (150 runs)** | 10 warm-up queries to prime caches + 50 queries executed across 3 randomized repetitions. |
| **Latency Metrics** | **P50, P95, P99, Mean** | Measured in milliseconds using high-resolution monotonic timer (`time.perf_counter_ns`). |
| **Throughput Metric** | **QPS** | Sequential Query Throughput (queries per second). |
| **IR Quality Metrics** | **Precision@10, Recall@10, Hit Rate@10, MRR, NDCG@10** | Evaluated against official SciFact ground-truth relevance judgments. |

---

## Empirical Experimental Results

The table below reflects genuine, experimentally measured numbers collected from the execution run:

| Metric / Parameter | Qdrant (Embedded) | Chroma (Persistent) | FAISS (In-Memory Baseline) | Milvus | Weaviate | pgvector | Elasticsearch | Pinecone |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Execution Status** | **BENCHMARKED** | **BENCHMARKED** | **BENCHMARKED** | *Not Benchmarked* | *Not Benchmarked* | *Not Benchmarked* | *Not Benchmarked* | *Not Benchmarked* |
| **Status Reason** | Local In-Process | Local Persistent | In-Process Baseline | *pymilvus uninstalled* | *weaviate uninstalled* | *no Postgres daemon* | *no ES daemon* | *API Key Not Set* |
| **Ingestion Throughput** | **1,121.7 vec/s** | **360.7 vec/s** | **422,493.6 vec/s** | — | — | — | — | — |
| **Total Ingestion Duration**| **0.892 s** | **2.772 s** | **0.002 s** | — | — | — | — | — |
| **Median Latency (P50)** | **4.43 ms** | **3.24 ms** | **0.23 ms** | — | — | — | — | — |
| **95th Percentile (P95)** | **6.70 ms** | **5.01 ms** | **0.47 ms** | — | — | — | — | — |
| **99th Percentile (P99)** | **7.42 ms** | **5.59 ms** | **1.73 ms** | — | — | — | — | — |
| **Mean Latency** | **4.77 ms** | **3.52 ms** | **0.28 ms** | — | — | — | — | — |
| **Throughput (QPS)** | **208.9 QPS** | **282.7 QPS** | **3,452.2 QPS** | — | — | — | — | — |
| **Recall@10** | **0.8700** | **0.8700** | **0.8700** | — | — | — | — | — |
| **Precision@10** | **0.0940** | **0.0940** | **0.0940** | — | — | — | — | — |
| **Hit Rate@10** | **0.8800** | **0.8800** | **0.8800** | — | — | — | — | — |
| **MRR** | **0.7295** | **0.7295** | **0.7295** | — | — | — | — | — |
| **NDCG@10** | **0.7602** | **0.7602** | **0.7602** | — | — | — | — | — |

### Empirical Insights
1. **Consistent Retrieval Behavior**: Qdrant, Chroma, and FAISS produced identical retrieval-quality scores under the controlled benchmark configuration, indicating consistent retrieval behavior for this workload.
2. **Algorithmic Library Ceiling vs Database Overhead**: FAISS operates strictly in memory as a C++ library without persistence, transaction logs, or network serialization, achieving 422,493.6 vectors/sec ingestion and 3,452.2 QPS with a 0.23 ms P50 latency.
3. **Embedded Database Durability**: Chroma (360.7 vec/s, 3.24 ms P50) and Qdrant (1,121.7 vec/s, 4.43 ms P50) deliver robust sub-5ms query response times while managing persistent metadata, payload indexing structures, and write-ahead logs.

---

## Technical Comparison of All 8 Vector Databases

The platform evaluates the 8 vector databases across key technical dimensions:

### 1. Architectural Taxonomy
- **Qdrant**: Native vector database engineered in Rust. Combines RocksDB metadata storage with segmented memory-mapped (`mmap`) vector payloads and segment write-ahead logs. Supports both embedded in-process execution and distributed multi-node clusters using Raft consensus.
- **Chroma**: Embedded / local-first vector store built in Python and Rust. Uses SQLite for metadata cataloging and schemas, coupled with `hnswlib` for vector index storage. Ideal for rapid prototyping and local LLM agents.
- **FAISS**: In-process C++ vector similarity-search library developed by Meta AI. Focuses strictly on algorithmic vector search and clustering (Flat, HNSW, IVF, PQ). Does not include server daemons, durability, CRUD mutations, or payload metadata filtering.
- **Milvus**: Distributed, cloud-native vector database (LF AI & Data Foundation). Fully decoupled architecture: stateless QueryNodes, DataNodes, and IndexNodes, backed by S3/MinIO object storage, Etcd coordination, and Apache Kafka/Pulsar log brokers acting as the write-ahead log.
- **Weaviate**: Native vector search engine written in Go. Uses a custom LSM-tree storage architecture for properties and vectors, dynamic HNSW index caches, built-in BM25 full-text search, and multi-node Raft consensus.
- **pgvector**: Open-source C extension for PostgreSQL by Andrew Kane. Brings HNSW and IVFFlat vector indexing directly into PostgreSQL relational tables, providing full ACID transaction guarantees, WAL durability, SQL query planner integration, and relational joins.
- **Elasticsearch**: Distributed search and analytics engine (Java / Apache Lucene). Dense vector indexing (`dense_vector`) using Lucene HNSW graphs alongside industry-standard Lucene inverted indexes for BM25 text relevance.
- **Pinecone**: Proprietary managed cloud vector database SaaS. Cloud-native serverless architecture decoupling query routing from tiered storage (NVMe SSD blob caches + cloud object storage) with automated index scaling.

### 2. Approximate Nearest Neighbor (ANN) Indexing
- **HNSW (Hierarchical Navigable Small World)**: Supported by Qdrant, Chroma, FAISS, Weaviate, pgvector, and Elasticsearch. Multi-layer proximity graph providing logarithmic $\mathcal{O}(\log N)$ search complexity.
- **Inverted File (IVF)**: Supported by FAISS and Milvus. Partitions the vector space into Voronoi cells via $k$-means clustering, probing only the closest centroids during search.
- **Vector Quantization (PQ / SQ / BQ)**:
  - *Scalar Quantization (SQ)*: Quantizes 32-bit floats to 8-bit integers (4x RAM reduction with minimal recall loss). Supported by Qdrant, Milvus, Weaviate, and Elasticsearch.
  - *Product Quantization (PQ)*: Decomposes vector dimensions into orthogonal sub-vectors and quantizes to centroids (8x–16x RAM reduction). Supported by FAISS, Milvus, Qdrant, and Weaviate.
  - *Binary Quantization (BQ)*: Thresholds vector components to 1-bit signs (32x compression for Hamming distance search). Supported by Qdrant, FAISS, Milvus, Weaviate, and pgvector (v0.7+).

### 3. Metadata Filtering Paradigms
- **Pre-Filtering**: Evaluates scalar filters first, then searches the remaining vectors. Can suffer from high latency when the filtered subset is small and graph connectivity is fragmented.
- **Post-Filtering**: Performs vector search across the entire dataset first, then discards non-matching hits. Suffers from recall collapse when filters are selective (Top-$K$ may return fewer than $K$ results).
- **Single-Stage (Integrated Graph Traversal)**: Implemented natively by Qdrant and Weaviate. Evaluates payload filter bitmasks dynamically at each hop of the HNSW graph traversal, guaranteeing exact Top-$K$ retrieval without recall loss.

### 4. Hybrid Search & Fusion
- **Reciprocal Rank Fusion (RRF)**:
  $$\text{RRF\_Score}(d) = \sum_{m \in M} \frac{1}{60 + \text{rank}_m(d)}$$
  Native support in Qdrant, Milvus, Weaviate, and Elasticsearch. Combines BM25 lexical keyword rankings with dense semantic vector rankings without requiring manual score normalization.

---

## Primary Excel Deliverable (`VDB_Comprehensive_Analysis.xlsx`)

The primary artifact strictly adheres to the requirement:
- **File**: [`outputs/excel/VDB_Comprehensive_Analysis.xlsx`](file:///c:/Users/User/Downloads/VED-Task/outputs/excel/VDB_Comprehensive_Analysis.xlsx)
- **Worksheet Count**: **EXACTLY 1** (named `VDB_Master_Comparison`)
- **Structure**: 78 total rows, 11 columns, frozen at cell `D6`
- **Visual Design**: Executive Navy banner (`#1B365D`), dark slate section dividers (`#243B53`), alternating clean white (`#FFFFFF`) and soft slate (`#F8FAFC`) striping, thin borders (`#CBD5E1`), and colored badges for `MEASURED` and `DOCUMENTATION` parameters.

### 14 Worksheet Sections:
- **Section A**: General Information & Metadata (Developer, Category, Core Language, License, Release Year, Documentation)
- **Section B**: Architecture & Storage Model (Deployment Topology, Storage Architecture, Decoupling, WAL & Durability)
- **Section C**: Vector Search & Indexing Capabilities (ANN Index Types, Exact kNN Flat, Distance Metrics, Quantization, Disk Indexing, Max Dimensions)
- **Section D**: Filtering & Hybrid Retrieval (Filtering Paradigms, Payload Indexing, BM25 Keyword Search, Sparse-Dense Fusion)
- **Section E**: Scalability & Distributed Architecture (Horizontal Sharding, Read Replicas, High Availability, Consensus Protocols)
- **Section F**: Developer Experience & Ecosystem (Client Protocols, Official SDKs, LangChain/LlamaIndex Integration)
- **Section G**: Deployment, Security & Operations (Embedded Mode, Docker/Kubernetes, RBAC & Authentication)
- **Section H**: Commercial & Cost Considerations (Cloud SaaS Availability, Infrastructure TCO)
- **Section I**: Benchmark Experimental Conditions (Execution Status, Corpus Docs, Queries, Judgments, Dimensions, Workload, Scope Notice, FAISS Scope)
- **Section J**: Measured Ingestion Performance (Vectors/Sec Throughput, Total Ingestion Duration, Index Build Time)
- **Section K**: Measured Query Latency & Throughput (Median P50, Tail P95, Tail P99, Mean Latency, Sequential QPS)
- **Section L**: Measured Information Retrieval (IR) Quality (Recall@10, Precision@10, Hit Rate@10, MRR, NDCG@10, Retrieval Quality Summary)
- **Section M**: Architectural Synthesis & Trade-offs (Core Advantages, Primary Architectural Limitations, Optimal Production Use Cases)
- **Section N**: Technical Documentation References (Official Documentation URLs, Official GitHub Repositories)

---

## Repository Structure

```
VED-Task/
├── README.md                           # Master technical documentation & empirical findings
├── requirements.txt                    # Pinned, tested Python dependencies
├── .gitignore                          # Standard git ignore rules (strictly excludes secrets/.env)
├── main.py                             # Unified CLI pipeline entry point
│
├── config/
│   └── benchmark_config.yaml           # Centralized configuration (dataset, model, VDBs)
│
├── data/
│   ├── raw/scifact/                    # Raw BEIR SciFact jsonl & qrels
│   ├── processed/                      # 1,000-doc & 50-query deterministic subsets
│   ├── embeddings/                     # Pre-computed 384-d numpy vector caches
│   └── models/                         # Local ONNX model weights and tokenizer
│
├── src/
│   ├── data/
│   │   └── beir_loader.py              # Download, extract, parse, and subsample SciFact
│   ├── embeddings/
│   │   └── embedder.py                 # ONNX MiniLM vector embedding engine
│   ├── adapters/
│   │   ├── base.py                     # BaseVectorDBAdapter interface definition
│   │   ├── qdrant_adapter.py           # Qdrant client implementation (embedded & client-server)
│   │   ├── chroma_adapter.py           # ChromaDB persistent client implementation
│   │   ├── faiss_adapter.py            # FAISS IndexFlatIP C++ library adapter
│   │   ├── milvus_adapter.py           # PyMilvus adapter with graceful fallback
│   │   ├── weaviate_adapter.py         # Weaviate v4 adapter with graceful fallback
│   │   ├── pgvector_adapter.py         # PostgreSQL pgvector adapter with fallback
│   │   ├── elasticsearch_adapter.py    # Elasticsearch Lucene dense_vector adapter
│   │   └── pinecone_adapter.py         # Pinecone Cloud SaaS adapter with fallback
│   ├── benchmark/
│   │   ├── runner.py                   # Latency, ingestion, and metric harness
│   │   ├── metrics.py                  # Recall@K, Precision@K, MRR, NDCG@K evaluators
│   │   ├── resource_monitor.py         # CPU, RSS RAM, and disk storage tracker
│   │   └── validator.py                # Health check and adapter smoke tester
│   ├── analysis/
│   │   └── comparison.py               # Exhaustive 30+ attribute knowledge matrix
│   ├── reporting/
│   │   └── excel_generator.py          # Single-worksheet openpyxl styling engine
│   └── utils/
│       └── logging_utils.py            # Formatted console and file logging
│
├── scripts/
│   ├── prepare_scifact.py              # Phase 1 standalone dataset script
│   ├── generate_embeddings.py          # Phase 2 standalone vector embedding script
│   ├── validate_adapters.py            # Phase 4 standalone adapter smoke test script
│   ├── run_benchmark.py                # Phase 5 standalone master benchmark script
│   └── generate_report.py              # Phase 8 standalone Excel generator script
│
└── outputs/
    ├── raw_results/
    │   └── benchmark_summary.json      # Structured JSON experimental results
    ├── logs/
    │   └── validation_report.json      # Smoke test health verification report
    └── excel/
        └── VDB_Comprehensive_Analysis.xlsx  # PRIMARY DELIVERABLE (Single Worksheet)
```

---

## Reproducibility & Execution Guide

### 1. Installation
```bash
python -m pip install -r requirements.txt
```

### 2. End-to-End Pipeline Execution
To execute data acquisition, embedding generation, adapter validation, benchmarking, and Excel reporting in one command:
```bash
python main.py --all
```

### 3. Modular Stage Execution
```bash
# Prepare dataset (BEIR/SciFact 1,000 docs / 50 queries)
python main.py --prepare-data

# Generate 384-dimensional unit-normalized embeddings via ONNX
python main.py --generate-embeddings

# Run adapter smoke test connectivity checks
python main.py --validate

# Run master benchmark across all databases
python main.py --run-benchmark

# Generate the single-worksheet Excel report
python main.py --generate-report
```

### 4. Excel Deliverable Verification
Verify that the generated workbook contains exactly one worksheet and opens cleanly:
```bash
python -c "import openpyxl; wb = openpyxl.load_workbook('outputs/excel/VDB_Comprehensive_Analysis.xlsx'); print('Sheets:', wb.sheetnames); print('Rows:', wb.active.max_row, 'Cols:', wb.active.max_column)"
```
*Expected Output:*
```
Sheets: ['VDB_Master_Comparison']
Rows: 78 Cols: 11
```

---

### Author & Attribution
Developed for the **AI/ML Engineering Technical Assessment** at **AgentAnalytics.AI**.
All source code, configuration matrices, benchmark engines, and Excel reports adhere to rigorous empirical standards.
