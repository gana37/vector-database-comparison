# Vector Database Comparative Analysis and Benchmarking Platform

An end-to-end engineering platform and empirical benchmarking framework evaluating **8 Vector Databases** on identical corpus data, embedding models, queries, and Information Retrieval (IR) metrics.

Built for the technical assessment requirement at **AgentAnalytics.AI**:
> *"1 Excel sheet, comparison analysis on 8 VDBs. Keep it exhaustive, technical depth."*

Primary Deliverable: [`outputs/excel/VDB_Comprehensive_Analysis.xlsx`](file:///c:/Users/User/Downloads/VED-Task/outputs/excel/VDB_Comprehensive_Analysis.xlsx) *(Strictly 1 single comprehensive worksheet: `VDB_Master_Comparison`)*

---

## Table of Contents
1. [Executive Summary & Architecture](#1-executive-summary--architecture)
2. [Evaluated Vector Database Systems](#2-evaluated-vector-database-systems)
3. [Scientific Benchmarking Methodology](#3-scientific-benchmarking-methodology)
4. [Empirical Experimental Results](#4-empirical-experimental-results)
5. [In-Depth Technical Comparison & Trade-off Analysis](#5-in-depth-technical-comparison--trade-off-analysis)
6. [Primary Excel Deliverable Specification](#6-primary-excel-deliverable-specification)
7. [Repository Structure](#7-repository-structure)
8. [Reproducibility & Execution Guide](#8-reproducibility--execution-guide)

---

## 1. Executive Summary & Architecture

Modern AI systems, Retrieval-Augmented Generation (RAG) pipelines, and autonomous agent platforms rely heavily on vector search engines for dense contextual retrieval. However, vector database selection is often driven by marketing claims rather than empirical validation and rigorous architectural trade-off analysis.

This platform provides:
- **Zero Fabrication**: Measured metrics are experimentally collected under controlled conditions. Unbenchmarked systems are explicitly identified with root-cause explanations (missing daemons, unconfigured API keys, or driver requirements).
- **Strict Vector Parity**: All systems receive identical unit-normalized 384-dimensional embeddings generated from the BEIR/SciFact benchmark.
- **Identical Evaluation Workload**: 1,000 corpus documents, 50 queries with 54 ground-truth relevance judgments, Top-$K=10$, 10 warmup queries, and 3 randomized measurement cycles (150 query runs per database).
- **Executive Deliverable**: An openpyxl-generated single-worksheet Excel workbook containing 73 rows across 14 sections spanning architecture, ANN algorithms, filtering, distributed scalability, developer experience, measured benchmarks, and production recommendations.

### Pipeline Architecture

```
                          ┌───────────────────────────┐
                          │   BEIR/SciFact Dataset    │
                          │   (TU Darmstadt Mirror)   │
                          └─────────────┬─────────────┘
                                        │ Download & Parse
                                        ▼
                          ┌───────────────────────────┐
                          │  Corpus (1,000 docs)      │
                          │  Queries (50 queries)     │
                          │  Qrels (54 judgments)     │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │  all-MiniLM-L6-v2 (ONNX)  │
                          │  384-d, Float32, L2-Norm  │
                          └──────┬─────────────┬──────┘
                                 │             │
        ┌────────────────────────┘             └────────────────────────┐
        ▼                                                               ▼
┌───────────────────────────────┐                             ┌───────────────────────────────┐
│ doc_embeddings_384d.npy (1000)│                             │query_embeddings_384d.npy (50) │
└───────────────┬───────────────┘                             └───────────────┬───────────────┘
                │                                                             │
                └───────────────────────┬─────────────────────────────────────┘
                                        │
                                        ▼
        ┌───────────────────────────────────────────────────────────────┐
        │                 Unified VDB Adapter Framework                 │
        │      (BaseVectorDBAdapter: connect, insert, query, cleanup)   │
        └───────┬──────────┬──────────┬──────────┬──────────┬───────────┘
                │          │          │          │          │
         ┌──────▼───┐ ┌────▼────┐ ┌───▼────┐ ┌───▼────┐ ┌───▼───────────┐
         │  Qdrant  │ │ Chroma  │ │ FAISS  │ │ Milvus │ │ Weaviate/etc. │
         └──────┬───┘ └────┬────┘ └───┬────┘ └───┬────┘ └───┬───────────┘
                │          │          │          │          │
                └──────────┴──────────┼──────────┴──────────┘
                                      │
                                      ▼
        ┌───────────────────────────────────────────────────────────────┐
        │            Benchmarking Engine & Resource Monitor             │
        │  - Ingestion Throughput (vec/s) & Index Construction Duration │
        │  - Latency Percentiles (P50, P90, P95, P99, Mean) & QPS       │
        │  - IR Quality (Recall@10, Precision@10, HitRate@10, MRR, NDCG)│
        └─────────────────────────────┬─────────────────────────────────┘
                                      │
                                      ▼
        ┌───────────────────────────────────────────────────────────────┐
        │                   Excel Report Generator                      │
        │      (openpyxl: Navy Palette, Freeze Panes, 1 Worksheet)      │
        └─────────────────────────────┬─────────────────────────────────┘
                                      │
                                      ▼
        ┌───────────────────────────────────────────────────────────────┐
        │         outputs/excel/VDB_Comprehensive_Analysis.xlsx         │
        └───────────────────────────────────────────────────────────────┘
```

---

## 2. Evaluated Vector Database Systems

The platform evaluates 8 vector search technologies representing the key archetypes in the vector database landscape:

| Database | Architecture Archetype | Core Engine | Primary ANN Index | License |
| :--- | :--- | :--- | :--- | :--- |
| **Qdrant** | Native Vector Database | Rust | HNSW (Payload-graph) | Apache 2.0 |
| **Chroma** | Embedded / Local-First Store | Python / Rust | HNSW (hnswlib) | Apache 2.0 |
| **FAISS** | Vector Similarity Library | C++ / CUDA | Flat, HNSW, IVF_PQ | MIT |
| **Milvus** | Distributed Cloud-Native VDB | Go / C++ | HNSW, DiskANN, SCANN | Apache 2.0 |
| **Weaviate** | Native Vector Search Engine | Go | Dynamic HNSW, Flat | BSD-3-Clause |
| **pgvector** | Relational DB Extension | C (PostgreSQL) | HNSW, IVFFlat | PostgreSQL |
| **Elasticsearch**| Enterprise Search Engine | Java (Lucene) | HNSW (dense_vector) | Elastic / SSPL |
| **Pinecone** | Managed Cloud SaaS | Proprietary | Hierarchical Graph | Proprietary |

---

## 3. Scientific Benchmarking Methodology

### A. Dataset Specification
- **Dataset**: `BEIR/SciFact` (Biomedical claim verification corpus with expert relevance judgments).
- **Corpus Subset**: Exactly 1,000 documents sampled deterministically (`seed=42`).
- **Query Subset**: Exactly 50 search queries with corresponding ground truth mappings.
- **Relevance Judgments (qrels)**: 54 binary positive judgments mapped within the subset.

### B. Embedding Generation & Parity
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`.
- **Inference Engine**: ONNX Runtime (`v1.30.0`) with HuggingFace Tokenizers (`v0.22.2`), bypassing Python GIL and CPython 3.14 free-threading DLL lockups.
- **Dimensionality**: 384 dimensions, `float32`.
- **Normalization**: $L_2$ unit normalization ($\|v\|_2 = 1.0$), ensuring Cosine Similarity equals Inner Product (Dot Product):
  $$\text{Cosine}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2} = u \cdot v \quad (\text{when } \|u\|=\|v\|=1)$$
- **Strict Parity**: Vectors were computed **once**, validated for NaN/infinite values, and saved to disk (`doc_embeddings_384d.npy` and `query_embeddings_384d.npy`). Every database ingested the identical numpy arrays.

### C. Experimental Rigor
- **Hardware Isolation**: Windows 10 x64, Python 3.14.2, AMD/Intel multicore CPU.
- **Warmup Cycles**: 10 warmup queries executed prior to latency recording to prime operating system caches and threadpools.
- **Measurement Repetitions**: 50 queries executed across 3 randomized repetitions (150 total queries per engine).
- **Latency Timer**: High-resolution monotonic clock (`time.perf_counter_ns`).

### D. Information Retrieval (IR) Evaluation Metrics
Given query $q$, retrieved top-$K$ candidate set $R_K(q)$, and ground-truth relevant set $G(q)$:

1. **Recall@K**:
   $$\text{Recall@}K = \frac{|R_K(q) \cap G(q)|}{|G(q)|}$$
2. **Precision@K**:
   $$\text{Precision@}K = \frac{|R_K(q) \cap G(q)|}{K}$$
3. **Hit Rate@K**:
   $$\text{Hit Rate@}K = \mathbb{I}(|R_K(q) \cap G(q)| > 0)$$
4. **Mean Reciprocal Rank (MRR)**:
   $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
5. **Normalized Discounted Cumulative Gain (NDCG@K)**:
   $$\text{DCG@}K = \sum_{i=1}^{K} \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}, \quad \text{NDCG@}K = \frac{\text{DCG@}K}{\text{IDCG@}K}$$

---

## 4. Empirical Experimental Results

All experiments were executed using identical 1,000 document vectors and 50 query vectors with $K=10$.

| Metric / Parameter | Qdrant (Embedded) | Chroma (Persistent) | FAISS (In-Memory Flat) | Milvus | Weaviate | pgvector | Elasticsearch | Pinecone |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Execution Status** | **BENCHMARKED** | **BENCHMARKED** | **BENCHMARKED** | Not Benchmarked | Not Benchmarked | Not Benchmarked | Not Benchmarked | Not Benchmarked |
| **Status Reason** | Local In-Process | Local Persistent | In-Process Baseline | *pymilvus uninstalled* | *weaviate uninstalled* | *no Postgres daemon* | *no ES daemon* | *API Key Not Set* |
| **Ingestion (vec/s)** | **797.0** | **386.4** | **300,571.1** | — | — | — | — | — |
| **Total Ingestion Time** | 1.255 s | 2.588 s | 0.003 s | — | — | — | — | — |
| **Median Latency (P50)** | **5.12 ms** | **4.85 ms** | **0.29 ms** | — | — | — | — | — |
| **95th Percentile (P95)** | **16.82 ms** | **11.05 ms** | **0.87 ms** | — | — | — | — | — |
| **99th Percentile (P99)** | 45.77 ms | 36.04 ms | 1.28 ms | — | — | — | — | — |
| **Mean Latency** | 7.40 ms | 6.53 ms | 0.37 ms | — | — | — | — | — |
| **Throughput (QPS)** | **134.2** | **152.6** | **2,590.7** | — | — | — | — | — |
| **Recall@10** | **0.8700** | **0.8700** | **0.8700** | — | — | — | — | — |
| **Precision@10** | **0.0940** | **0.0940** | **0.0940** | — | — | — | — | — |
| **Hit Rate@10** | **0.8800** | **0.8800** | **0.8800** | — | — | — | — | — |
| **MRR** | **0.7295** | **0.7295** | **0.7295** | — | — | — | — | — |
| **NDCG@10** | **0.7602** | **0.7602** | **0.7602** | — | — | — | — | — |

### Key Experimental Insights
1. **Mathematical Quality Parity**: Qdrant, Chroma, and FAISS achieved **identical** retrieval quality scores ($\text{Recall@10} = 0.8700$, $\text{MRR} = 0.7295$, $\text{NDCG@10} = 0.7602$). This proves that embedding normalization and index distance math were strictly equivalent across all three engines.
2. **Raw Vector Compute vs Database Overhead**: FAISS operates purely as an in-memory C++ library without transaction logs, locking, or persistence overhead, achieving 300,571 vectors/sec ingestion and 2,590.7 QPS with sub-millisecond P50 ($0.29\text{ ms}$).
3. **Database Durability Trade-offs**: Chroma and Qdrant incorporate persistent metadata storage, payload tracking, and search indexing structures, delivering practical production query latencies ($4.85\text{ ms}$ and $5.12\text{ ms}$ P50) while providing persistent storage.

---

## 5. In-Depth Technical Comparison & Trade-off Analysis

### 1. Storage & Persistence Architectures
- **Qdrant**: Employs RocksDB and memory-mapped (`mmap`) segment vector payloads. Mutations append to a Write-Ahead Log (WAL) with configurable segment optimization.
- **Chroma**: Utilizes SQLite for collection and metadata schemas, with HNSW graph vector storage serialized through `hnswlib`.
- **FAISS**: Pure in-memory C++ contiguous buffers. Persistence is manual (explicit file serialization) or via memory-mapped IVF inverted lists (`OnDiskInvertedLists`).
- **Milvus**: Cloud-native decoupled architecture. Log broker (Kafka/Pulsar) serves as WAL; immutable data segments are stored in S3/MinIO; state is coordinated via Etcd.
- **Weaviate**: Custom LSM-tree architecture for property storage with inverted indexes and dynamic HNSW vector caches.
- **pgvector**: Extends PostgreSQL heap pages and shared buffer cache. Fully participates in PostgreSQL ACID transactions and Write-Ahead Logging (WAL).
- **Elasticsearch**: Apache Lucene immutable segments with segment merging and translog durability. Dense vectors are stored in `.vec` and `.vem` Lucene files.
- **Pinecone**: Proprietary managed cloud storage tiered between ultra-fast NVMe SSD blob caches and object storage.

### 2. Approximate Nearest Neighbor (ANN) Indexing
- **HNSW (Hierarchical Navigable Small World)**: Used by Qdrant, Chroma, FAISS, Weaviate, pgvector, and Elasticsearch. Constructs a multi-layer geometric graph enabling logarithmic search complexity $\mathcal{O}(\log N)$.
- **Inverted File (IVF)**: Supported by FAISS and Milvus. Partitions vector space into Voronoi cells via $k$-means clustering, searching only candidate cells during query time.
- **Quantization (PQ / SQ / BQ)**:
  - *Scalar Quantization (SQ)*: Compresses 32-bit floats to 8-bit integers (4x memory reduction).
  - *Product Quantization (PQ)*: Decomposes vector dimensions into sub-vectors and quantizes into centroids (8x–16x memory reduction).
  - *Binary Quantization (BQ)*: Thresholds vector components to 1-bit signs (32x memory reduction).

### 3. Metadata Filtering Paradigms
- **Pre-Filtering**: Filters candidate IDs using scalar/relational conditions first, then performs vector search on the remaining subset. Suffers from high latency when the filtered subset is small and graph connectivity is broken.
- **Post-Filtering**: Performs ANN search on the entire vector space first, then discards hits that violate metadata filters. Suffers from recall collapse when filters are highly selective (top-$K$ may return 0 matches).
- **Single-Stage (Integrated Graph Traversal)**: Implemented natively by Qdrant and Weaviate. Traverses the HNSW graph while dynamically evaluating payload condition bitmasks at each vertex hop, guaranteeing exact Top-$K$ retrieval without recall loss.

### 4. Hybrid Search (Dense + Sparse Fusion)
- **Reciprocal Rank Fusion (RRF)**:
  $$\text{RRF\_Score}(d) = \sum_{m \in M} \frac{1}{60 + \text{rank}_m(d)}$$
  Supported natively by Qdrant, Milvus, Weaviate, and Elasticsearch. Combines BM25 lexical keyword scores with dense vector semantic rankings without requiring score normalization.

---

## 6. Primary Excel Deliverable Specification

The generated workbook strictly meets all requirements:
- **File Path**: [`outputs/excel/VDB_Comprehensive_Analysis.xlsx`](file:///c:/Users/User/Downloads/VED-Task/outputs/excel/VDB_Comprehensive_Analysis.xlsx)
- **Worksheet Count**: **EXACTLY 1** (named `VDB_Master_Comparison`)
- **Total Columns**: 11 (`Technical Parameter`, `Dimension / Category`, `Data Type`, and the 8 VDBs)
- **Total Rows**: 73 structured rows

### Worksheet Layout & Hierarchy
```
Row 1:  [MASTER BANNER] Deep Navy (#1B365D), Bold White 14pt (A1:K1 merged)
Row 2:  [SUBTITLE] Slate Navy (#2C3E50), Italic White 10pt (A2:K2 merged)
Row 3:  [METADATA] Soft Slate Blue (#EAEFF5), Bold 9pt (A3:K3 merged)
Row 4:  [SPACER ROW]
Row 5:  [TABLE HEADERS] Navy Fill (#1B365D), Bold White 10.5pt, Centered
Row 6+: [SECTIONS & DATA ROWS]
        ├── SECTION A: General Information & Metadata (5 rows)
        ├── SECTION B: Architecture & Storage Model (4 rows)
        ├── SECTION C: Vector Search & Indexing Capabilities (6 rows)
        ├── SECTION D: Filtering & Hybrid Retrieval (4 rows)
        ├── SECTION E: Scalability & Distributed Architecture (3 rows)
        ├── SECTION F: Developer Experience & Ecosystem (3 rows)
        ├── SECTION G: Deployment, Security & Operations (3 rows)
        ├── SECTION H: Commercial & Cost Considerations (2 rows)
        ├── SECTION I: Benchmark Experimental Conditions (5 rows)
        ├── SECTION J: Measured Ingestion Performance (3 rows)
        ├── SECTION K: Measured Query Latency & Throughput (5 rows)
        ├── SECTION L: Measured Information Retrieval (IR) Quality (5 rows)
        ├── SECTION M: Architectural Synthesis & Trade-offs (3 rows)
        └── SECTION N: Technical Documentation References (2 rows)
```

### Visual & Functional Design Standards
- **Freeze Panes**: Frozen at cell `D6`. Headers (Rows 1–5) and Parameter Identifiers (Columns A–C) remain anchored during horizontal and vertical scrolling.
- **Section Dividers**: Deep Blue-Gray (`#243B53`), white bold text, row height 24pt.
- **Zebra Striping**: Alternating clean white (`#FFFFFF`) and slate tint (`#F8FAFC`).
- **Data Badging**:
  - `MEASURED`: Forest green fill (`#ECFDF5`), emerald text (`#065F46`).
  - `DOCUMENTATION`: Soft blue fill (`#EFF6FF`), royal blue text (`#1E40AF`).
- **Measured Data Formatting**: Benchmarked cells styled with subtle green background (`#F0FDF4`); unbenchmarked cells styled in muted italic slate (`#64748B`).

---

## 7. Repository Structure

```
VED-Task/
├── README.md                           # Master architectural and benchmark documentation
├── requirements.txt                    # Python package dependencies
├── .gitignore                          # Standard git exclusion rules
├── main.py                             # Master CLI orchestration pipeline
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
│   │   ├── qdrant_adapter.py           # Qdrant client implementation
│   │   ├── chroma_adapter.py           # ChromaDB persistent client implementation
│   │   ├── faiss_adapter.py            # FAISS IndexFlatIP C++ engine adapter
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

## 8. Reproducibility & Execution Guide

### Prerequisites
- Python 3.10+ (Tested on Python 3.14.2 on Windows 10 x64)
- Git and Internet connection (for initial SciFact download)

### Step 1: Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### Step 2: Run End-to-End Pipeline
To run data acquisition, embedding generation, adapter validation, benchmarking, and Excel report generation in a single automated command:
```bash
python main.py --all
```

### Step 3: Run Modular Stages Individually
```bash
# 1. Download SciFact and generate 1,000 doc / 50 query deterministic subset
python main.py --prepare-data

# 2. Generate 384-dimensional unit-normalized embeddings via ONNX Runtime
python main.py --generate-embeddings

# 3. Perform smoke test connectivity checks on all 8 database adapters
python main.py --validate

# 4. Execute the master benchmark harness across all databases
python main.py --run-benchmark

# 5. Generate the final single-worksheet Excel comparison report
python main.py --generate-report
```

### Step 4: Verify the Excel Deliverable
```bash
python -c "import openpyxl; wb = openpyxl.load_workbook('outputs/excel/VDB_Comprehensive_Analysis.xlsx'); print('Sheets:', wb.sheetnames); print('Total Rows:', wb.active.max_row); print('Total Columns:', wb.active.max_column)"
```
**Expected Verification Output:**
```
Sheets: ['VDB_Master_Comparison']
Total Rows: 73
Total Columns: 11
```

---

### Author & Attribution
Developed for the **AI/ML Engineering Technical Assessment** at **AgentAnalytics.AI**.
All source code, configuration matrices, benchmark engines, and Excel reports adhere to rigorous empirical standards.
