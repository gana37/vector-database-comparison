"""
Weaviate Adapter: Native Vector Search Engine (Go/C++ core).
Supports GraphQL/gRPC client-server architectures with modular vectorizers.
"""
import time
import socket
import numpy as np
from typing import Dict, List, Any, Optional
from src.adapters.base import BaseVectorDBAdapter, SearchResult, HealthCheckResult
from src.utils.logging_utils import setup_logger

logger = setup_logger("weaviate_adapter")

class WeaviateAdapter(BaseVectorDBAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Weaviate", config)
        self.host = config.get("host", "localhost")
        self.port = int(config.get("port", 8080))
        self.grpc_port = int(config.get("grpc_port", 50051))
        self.class_name = config.get("class_name", "SciFactDocument")
        self.client = None

    def _is_port_open(self) -> bool:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1.0)
            res = s.connect_ex((self.host, self.port))
            s.close()
            return res == 0
        except Exception:
            return False

    def health_check(self) -> HealthCheckResult:
        try:
            import weaviate
            version = getattr(weaviate, "__version__", "4.x")
        except ImportError:
            return HealthCheckResult(
                is_available=False,
                status_message="weaviate-client package not installed.",
                deployment_mode="Native Vector Search Engine"
            )

        if not self._is_port_open():
            return HealthCheckResult(
                is_available=False,
                status_message=f"Weaviate daemon not reachable on {self.host}:{self.port}. Requires running Docker container (cr.weaviate.io/semitechnologies/weaviate).",
                version=version,
                deployment_mode="Native Vector Search Engine (Server)"
            )

        try:
            import weaviate
            client = weaviate.connect_to_local(host=self.host, port=self.port, grpc_port=self.grpc_port)
            ready = client.is_ready()
            client.close()
            return HealthCheckResult(
                is_available=ready,
                status_message="Weaviate service is ready." if ready else "Weaviate service not ready.",
                version=version,
                deployment_mode="Native Vector Search Engine (Server)"
            )
        except Exception as e:
            return HealthCheckResult(
                is_available=False,
                status_message=f"Weaviate connection failed: {e}",
                version=version,
                deployment_mode="Native Vector Search Engine (Server)"
            )

    def connect(self) -> None:
        import weaviate
        self.client = weaviate.connect_to_local(host=self.host, port=self.port, grpc_port=self.grpc_port)
        self.is_connected = True

    def create_collection(self, collection_name: str, dimension: int, metric: str = "cosine") -> None:
        self.connect()
        import weaviate.classes.config as wvc
        
        self.class_name = collection_name
        if self.client.collections.exists(collection_name):
            self.client.collections.delete(collection_name)

        self.client.collections.create(
            name=collection_name,
            vectorizer_config=wvc.Configure.Vectorizer.none(),
            vector_index_config=wvc.Configure.VectorIndex.hnsw(
                distance_metric=wvc.VectorDistances.COSINE,
                max_connections=16,
                ef_construction=100
            ),
            properties=[
                wvc.Property(name="doc_id", data_type=wvc.DataType.TEXT),
                wvc.Property(name="title", data_type=wvc.DataType.TEXT),
            ]
        )

    def insert(self, vectors: np.ndarray, payloads: List[Dict[str, Any]], ids: List[str]) -> float:
        if self.client is None:
            raise RuntimeError("Client not connected.")

        t0 = time.perf_counter()
        collection = self.client.collections.get(self.class_name)
        with collection.batch.dynamic() as batch:
            for vec, did, payload in zip(vectors, ids, payloads if payloads else [{}] * len(ids)):
                batch.add_object(
                    properties={"doc_id": str(did), "title": payload.get("title", "")},
                    vector=vec.tolist()
                )
        return time.perf_counter() - t0

    def build_index(self, index_params: Optional[Dict[str, Any]] = None) -> float:
        return 0.0

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        if self.client is None:
            raise RuntimeError("Client not connected.")

        collection = self.client.collections.get(self.class_name)
        response = collection.query.near_vector(
            near_vector=query_vector.tolist(),
            limit=top_k,
            return_metadata=["distance"]
        )

        results = []
        for obj in response.objects:
            doc_id = str(obj.properties.get("doc_id", obj.uuid))
            dist = obj.metadata.distance if obj.metadata and obj.metadata.distance else 0.0
            score = 1.0 - float(dist)
            results.append(SearchResult(doc_id=doc_id, score=score, payload=dict(obj.properties)))
        return results

    def get_storage_size(self) -> Optional[int]:
        return None

    def cleanup(self) -> None:
        if self.client is not None:
            try:
                if self.client.collections.exists(self.class_name):
                    self.client.collections.delete(self.class_name)
                self.client.close()
            except Exception:
                pass
            self.client = None
            self.is_connected = False
