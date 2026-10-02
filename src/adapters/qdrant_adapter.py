"""
Qdrant Adapter: Native Vector Database (Rust Core).
Supports embedded local persistent storage and client-server modes.
"""
import time
import os
import shutil
import numpy as np
from typing import Dict, List, Any, Optional
from src.adapters.base import BaseVectorDBAdapter, SearchResult, HealthCheckResult
from src.utils.logging_utils import setup_logger

logger = setup_logger("qdrant_adapter")

class QdrantAdapter(BaseVectorDBAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Qdrant", config)
        self.mode = config.get("mode", "embedded")
        self.storage_path = config.get("path", "data/qdrant_storage")
        self.hnsw_m = config.get("hnsw_m", 16)
        self.ef_construct = config.get("ef_construct", 100)
        self.client = None
        self.models = None

    def health_check(self) -> HealthCheckResult:
        try:
            import qdrant_client
            from qdrant_client import models
            self.models = models
            import importlib.metadata
            version = importlib.metadata.version("qdrant-client")
            return HealthCheckResult(
                is_available=True,
                status_message=f"Qdrant client operational in {self.mode} mode.",
                version=version,
                deployment_mode=f"Native Vector DB ({self.mode.capitalize()})"
            )
        except Exception as e:
            return HealthCheckResult(
                is_available=False,
                status_message=f"Qdrant initialization error: {e}",
                deployment_mode="Native Vector DB"
            )

    def connect(self) -> None:
        if self.client is None:
            import qdrant_client
            from qdrant_client import models
            self.models = models
            if self.mode == "embedded":
                if self.storage_path == ":memory:":
                    self.client = qdrant_client.QdrantClient(":memory:")
                else:
                    os.makedirs(self.storage_path, exist_ok=True)
                    self.client = qdrant_client.QdrantClient(path=self.storage_path)
            else:
                host = self.config.get("host", "localhost")
                port = self.config.get("port", 6333)
                self.client = qdrant_client.QdrantClient(host=host, port=port)
            self.is_connected = True

    def create_collection(self, collection_name: str, dimension: int, metric: str = "cosine") -> None:
        self.connect()
        self.collection_name = collection_name
        self.dimension = dimension
        
        # Map distance metric
        distance_map = {
            "cosine": self.models.Distance.COSINE,
            "dot": self.models.Distance.DOT,
            "euclidean": self.models.Distance.EUCLID
        }
        distance = distance_map.get(metric.lower(), self.models.Distance.COSINE)

        # Recreate collection
        if self.client.collection_exists(collection_name):
            self.client.delete_collection(collection_name)

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=self.models.VectorParams(size=dimension, distance=distance),
            hnsw_config=self.models.HnswConfigDiff(
                m=self.hnsw_m,
                ef_construct=self.ef_construct
            )
        )

    def insert(self, vectors: np.ndarray, payloads: List[Dict[str, Any]], ids: List[str]) -> float:
        if self.client is None:
            raise RuntimeError("Client not connected.")

        t0 = time.perf_counter()
        batch_size = 256
        total = len(ids)

        for i in range(0, total, batch_size):
            batch_vectors = vectors[i : i + batch_size].tolist()
            batch_payloads = payloads[i : i + batch_size] if payloads else [{}] * len(batch_vectors)
            batch_ids = ids[i : i + batch_size]

            points = [
                self.models.PointStruct(
                    id=idx + i, # numeric point ID for internal efficiency
                    vector=vec,
                    payload={"doc_id": str(doc_id), **pay}
                )
                for idx, (vec, doc_id, pay) in enumerate(zip(batch_vectors, batch_ids, batch_payloads))
            ]
            self.client.upsert(collection_name=self.collection_name, points=points, wait=True)

        duration = time.perf_counter() - t0
        return duration

    def build_index(self, index_params: Optional[Dict[str, Any]] = None) -> float:
        # In Qdrant, HNSW index segments build continuously in the background
        return 0.0

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        if self.client is None:
            raise RuntimeError("Client not connected.")

        q_list = query_vector.tolist()
        
        # Native query
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=q_list,
            limit=top_k,
            with_payload=True
        )

        results = []
        for point in search_result.points:
            doc_id = str(point.payload.get("doc_id", point.id))
            score = float(point.score)
            results.append(SearchResult(doc_id=doc_id, score=score, payload=point.payload))

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
        if self.client is not None:
            try:
                if self.client.collection_exists(self.collection_name):
                    self.client.delete_collection(self.collection_name)
            except Exception:
                pass
            self.client = None
            self.is_connected = False
        if os.path.exists(self.storage_path):
            try:
                shutil.rmtree(self.storage_path, ignore_errors=True)
            except Exception:
                pass
