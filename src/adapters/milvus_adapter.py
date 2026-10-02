"""
Milvus Adapter: Distributed Cloud-Native Vector Database (Go/C++ core).
Supports standalone server, distributed cluster, or milvus-lite.
"""
import time
import socket
import numpy as np
from typing import Dict, List, Any, Optional
from src.adapters.base import BaseVectorDBAdapter, SearchResult, HealthCheckResult
from src.utils.logging_utils import setup_logger

logger = setup_logger("milvus_adapter")

class MilvusAdapter(BaseVectorDBAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Milvus", config)
        self.host = config.get("host", "localhost")
        self.port = int(config.get("port", 19530))
        self.index_type = config.get("index_type", "HNSW")
        self.metric_type = config.get("metric_type", "COSINE")
        self.client = None
        self.collection = None

    def _is_port_open(self) -> bool:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1.0)
            result = s.connect_ex((self.host, self.port))
            s.close()
            return result == 0
        except Exception:
            return False

    def health_check(self) -> HealthCheckResult:
        try:
            import pymilvus
            version = getattr(pymilvus, "__version__", "2.4.x")
        except ImportError:
            return HealthCheckResult(
                is_available=False,
                status_message="pymilvus package not installed.",
                deployment_mode="Distributed Vector Database"
            )

        # Check socket connectivity to Milvus standalone / cluster daemon
        if not self._is_port_open():
            return HealthCheckResult(
                is_available=False,
                status_message=f"Milvus daemon not reachable on {self.host}:{self.port}. Requires running Docker container (milvus-standalone).",
                version=version,
                deployment_mode="Distributed Cloud-Native Vector Database"
            )

        try:
            from pymilvus import connections
            connections.connect(alias="default", host=self.host, port=self.port, timeout=2.0)
            connections.disconnect("default")
            return HealthCheckResult(
                is_available=True,
                status_message=f"Connected successfully to Milvus at {self.host}:{self.port}.",
                version=version,
                deployment_mode="Distributed Cloud-Native Vector Database"
            )
        except Exception as e:
            return HealthCheckResult(
                is_available=False,
                status_message=f"Milvus connection failed: {e}",
                version=version,
                deployment_mode="Distributed Cloud-Native Vector Database"
            )

    def connect(self) -> None:
        from pymilvus import connections
        connections.connect(alias="default", host=self.host, port=self.port)
        self.is_connected = True

    def create_collection(self, collection_name: str, dimension: int, metric: str = "cosine") -> None:
        self.connect()
        from pymilvus import FieldSchema, CollectionSchema, DataType, Collection, utility

        self.collection_name = collection_name
        self.dimension = dimension

        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension)
        ]
        schema = CollectionSchema(fields=fields, description="SciFact benchmark collection")
        self.collection = Collection(name=collection_name, schema=schema)

    def insert(self, vectors: np.ndarray, payloads: List[Dict[str, Any]], ids: List[str]) -> float:
        if self.collection is None:
            raise RuntimeError("Collection not initialized.")

        t0 = time.perf_counter()
        int_ids = list(range(len(ids)))
        str_ids = [str(did) for did in ids]
        data = [
            int_ids,
            str_ids,
            vectors.tolist()
        ]
        self.collection.insert(data)
        self.collection.flush()
        return time.perf_counter() - t0

    def build_index(self, index_params: Optional[Dict[str, Any]] = None) -> float:
        if self.collection is None:
            raise RuntimeError("Collection not initialized.")

        t0 = time.perf_counter()
        index_spec = {
            "metric_type": "COSINE" if self.metric.lower() == "cosine" else "IP",
            "index_type": self.index_type,
            "params": {"M": 16, "efConstruction": 100}
        }
        self.collection.create_index(field_name="vector", index_params=index_spec)
        self.collection.load()
        return time.perf_counter() - t0

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        if self.collection is None:
            raise RuntimeError("Collection not initialized.")

        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
        res = self.collection.search(
            data=[query_vector.tolist()],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            output_fields=["doc_id"]
        )

        results = []
        for hit in res[0]:
            doc_id = str(hit.entity.get("doc_id"))
            score = float(hit.score)
            results.append(SearchResult(doc_id=doc_id, score=score))
        return results

    def get_storage_size(self) -> Optional[int]:
        return None

    def cleanup(self) -> None:
        if self.collection is not None:
            try:
                from pymilvus import utility
                if utility.has_collection(self.collection_name):
                    utility.drop_collection(self.collection_name)
            except Exception:
                pass
            self.collection = None
            self.is_connected = False
