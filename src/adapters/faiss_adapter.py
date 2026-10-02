"""
FAISS Adapter: In-process Algorithmic Vector Search Library (Meta AI).
Acts as the baseline exact/ANN search benchmark.
"""
import time
import os
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from src.adapters.base import BaseVectorDBAdapter, SearchResult, HealthCheckResult
from src.utils.logging_utils import setup_logger

logger = setup_logger("faiss_adapter")

class FAISSAdapter(BaseVectorDBAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__("FAISS", config)
        self.index = None
        self.id_to_doc_id: Dict[int, str] = {}
        self.doc_id_to_id: Dict[str, int] = {}
        self.payloads: Dict[str, Dict[str, Any]] = {}
        self.index_type = config.get("index_type", "IndexFlatIP")
        self.hnsw_m = config.get("hnsw_m", 16)
        self.faiss_module = None

    def health_check(self) -> HealthCheckResult:
        try:
            import faiss
            self.faiss_module = faiss
            version = getattr(faiss, "__version__", "1.13.2")
            return HealthCheckResult(
                is_available=True,
                status_message="FAISS library is installed and operational (In-Process CPU).",
                version=version,
                deployment_mode="In-Process Algorithmic Library"
            )
        except Exception as e:
            return HealthCheckResult(
                is_available=False,
                status_message=f"FAISS import failed: {e}",
                deployment_mode="In-Process Algorithmic Library"
            )

    def connect(self) -> None:
        if self.faiss_module is None:
            import faiss
            self.faiss_module = faiss
        self.is_connected = True

    def create_collection(self, collection_name: str, dimension: int, metric: str = "cosine") -> None:
        self.connect()
        self.dimension = dimension
        self.metric = metric
        self.id_to_doc_id.clear()
        self.doc_id_to_id.clear()
        self.payloads.clear()

        # For unit-normalized vectors: Inner Product == Cosine Similarity
        if self.index_type == "HNSW":
            self.index = self.faiss_module.IndexHNSWFlat(dimension, self.hnsw_m, self.faiss_module.METRIC_INNER_PRODUCT)
            self.index.hnsw.efSearch = 64
            self.index.hnsw.efConstruction = 100
        else:
            # Default: Exact brute-force IndexFlatIP
            self.index = self.faiss_module.IndexFlatIP(dimension)

    def insert(self, vectors: np.ndarray, payloads: List[Dict[str, Any]], ids: List[str]) -> float:
        if self.index is None:
            raise RuntimeError("Index not initialized. Call create_collection first.")

        t0 = time.perf_counter()
        vectors_f32 = np.ascontiguousarray(vectors, dtype=np.float32)
        start_idx = self.index.ntotal
        
        self.index.add(vectors_f32)
        
        for idx, doc_id in enumerate(ids):
            int_id = start_idx + idx
            self.id_to_doc_id[int_id] = str(doc_id)
            self.doc_id_to_id[str(doc_id)] = int_id
            if idx < len(payloads):
                self.payloads[str(doc_id)] = payloads[idx]
                
        duration = time.perf_counter() - t0
        return duration

    def build_index(self, index_params: Optional[Dict[str, Any]] = None) -> float:
        # FAISS IndexFlatIP requires no training; HNSW builds incrementally on add
        return 0.0

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        if self.index is None:
            raise RuntimeError("Index not initialized.")

        q_vec = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        
        # FAISS native search
        scores, indices = self.index.search(q_vec, top_k)
        
        results = []
        for rank in range(len(indices[0])):
            int_id = int(indices[0][rank])
            if int_id == -1 or int_id not in self.id_to_doc_id:
                continue
            doc_id = self.id_to_doc_id[int_id]
            score = float(scores[0][rank])
            payload = self.payloads.get(doc_id, {})
            
            # Simple payload filter post-check if requested
            if filter_criteria:
                match = all(payload.get(k) == v for k, v in filter_criteria.items())
                if not match:
                    continue
                    
            results.append(SearchResult(doc_id=doc_id, score=score, payload=payload))
            if len(results) >= top_k:
                break
                
        return results

    def get_storage_size(self) -> Optional[int]:
        if self.index is not None:
            # Vector raw memory: ntotal * dimension * 4 bytes
            return int(self.index.ntotal * self.dimension * 4)
        return None

    def cleanup(self) -> None:
        self.index = None
        self.id_to_doc_id.clear()
        self.doc_id_to_id.clear()
        self.payloads.clear()
