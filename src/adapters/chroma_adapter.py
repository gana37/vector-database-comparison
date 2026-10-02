"""
ChromaDB Adapter: Local-First / Embedded Vector Store (Python/Rust backend).
Runs in-process via chromadb.PersistentClient with HNSW indexing.
"""
import time
import os
import shutil
import numpy as np
from typing import Dict, List, Any, Optional
from src.adapters.base import BaseVectorDBAdapter, SearchResult, HealthCheckResult
from src.utils.logging_utils import setup_logger

logger = setup_logger("chroma_adapter")

class ChromaAdapter(BaseVectorDBAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Chroma", config)
        self.mode = config.get("mode", "embedded")
        self.storage_path = config.get("path", "data/chroma_storage")
        self.client = None
        self.collection = None

    def health_check(self) -> HealthCheckResult:
        try:
            import chromadb
            version = getattr(chromadb, "__version__", "1.5.9")
            return HealthCheckResult(
                is_available=True,
                status_message="ChromaDB persistent client is operational (Embedded HNSW).",
                version=version,
                deployment_mode="Embedded / Local-First Vector Store"
            )
        except Exception as e:
            return HealthCheckResult(
                is_available=False,
                status_message=f"ChromaDB initialization error: {e}",
                deployment_mode="Embedded Vector Store"
            )

    def connect(self) -> None:
        if self.client is None:
            import chromadb
            from chromadb.config import Settings
            os.environ["ANONYMIZED_TELEMETRY"] = "False"
            os.makedirs(self.storage_path, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=self.storage_path,
                settings=Settings(anonymized_telemetry=False, is_persistent=True)
            )
            self.is_connected = True

    def create_collection(self, collection_name: str, dimension: int, metric: str = "cosine") -> None:
        self.connect()
        self.collection_name = collection_name
        self.dimension = dimension

        # Chroma spaces: "cosine", "l2", "ip"
        space = "cosine" if metric.lower() == "cosine" else "ip"
        
        # Reset collection if exists
        try:
            self.client.delete_collection(name=collection_name)
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": space},
            embedding_function=None # Disable automatic internal embedding
        )

    def insert(self, vectors: np.ndarray, payloads: List[Dict[str, Any]], ids: List[str]) -> float:
        if self.collection is None:
            raise RuntimeError("Collection not initialized.")

        t0 = time.perf_counter()
        batch_size = 256
        total = len(ids)

        for i in range(0, total, batch_size):
            batch_vectors = vectors[i : i + batch_size].tolist()
            batch_ids = [str(did) for did in ids[i : i + batch_size]]
            
            # Format metadata
            batch_metadatas = []
            for did, p in zip(batch_ids, payloads[i : i + batch_size] if payloads else [{}] * len(batch_ids)):
                clean_meta = {"doc_id": str(did)}
                for k, v in p.items():
                    if isinstance(v, (str, int, float, bool)):
                        clean_meta[k] = v
                batch_metadatas.append(clean_meta)

            self.collection.add(
                ids=batch_ids,
                embeddings=batch_vectors,
                metadatas=batch_metadatas
            )

        duration = time.perf_counter() - t0
        return duration

    def build_index(self, index_params: Optional[Dict[str, Any]] = None) -> float:
        # Chroma builds HNSW graphs incrementally during insertion
        return 0.0

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        if self.collection is None:
            raise RuntimeError("Collection not initialized.")

        q_list = query_vector.tolist()
        
        where_filter = filter_criteria if filter_criteria else None
        query_res = self.collection.query(
            query_embeddings=[q_list],
            n_results=top_k,
            where=where_filter,
            include=["metadatas", "distances"]
        )

        results = []
        if query_res and query_res["ids"] and len(query_res["ids"][0]) > 0:
            retrieved_ids = query_res["ids"][0]
            retrieved_distances = query_res["distances"][0] if "distances" in query_res else [0.0] * len(retrieved_ids)
            retrieved_metas = query_res["metadatas"][0] if "metadatas" in query_res else [{}] * len(retrieved_ids)

            for did, dist, meta in zip(retrieved_ids, retrieved_distances, retrieved_metas):
                # For cosine space in Chroma: score = 1.0 - distance
                score = 1.0 - float(dist)
                results.append(SearchResult(doc_id=str(did), score=score, payload=meta))

        return results

    def get_storage_size(self) -> Optional[int]:
        if os.path.exists(self.storage_path):
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(self.storage_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if os.path.exists(fp):
                        total_size += os.path.getsize(fp)
            return total_size
        return None

    def cleanup(self) -> None:
        if self.client is not None and self.collection is not None:
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass
            self.collection = None
            self.client = None
            self.is_connected = False
        if os.path.exists(self.storage_path):
            try:
                shutil.rmtree(self.storage_path, ignore_errors=True)
            except Exception:
                pass
