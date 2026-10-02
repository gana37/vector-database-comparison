"""
Elasticsearch Adapter: Search Engine with Dense Vector Capabilities (Apache Lucene core).
Supports exact kNN, HNSW dense_vector indexing, and hybrid BM25 search.
"""
import time
import requests
import numpy as np
from typing import Dict, List, Any, Optional
from src.adapters.base import BaseVectorDBAdapter, SearchResult, HealthCheckResult
from src.utils.logging_utils import setup_logger

logger = setup_logger("elasticsearch_adapter")

class ElasticsearchAdapter(BaseVectorDBAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Elasticsearch", config)
        self.host = config.get("host", "http://localhost:9200")
        self.index_name = config.get("index_name", "scifact_benchmark")
        self.client = None

    def health_check(self) -> HealthCheckResult:
        try:
            import elasticsearch
            version = getattr(elasticsearch, "__version__", "8.x")
        except ImportError:
            return HealthCheckResult(
                is_available=False,
                status_message="elasticsearch python client not installed.",
                deployment_mode="Enterprise Search Engine (Lucene)"
            )

        try:
            r = requests.get(f"{self.host}", timeout=2.0)
            if r.status_code == 200:
                data = r.json()
                es_version = data.get("version", {}).get("number", "8.x")
                return HealthCheckResult(
                    is_available=True,
                    status_message=f"Elasticsearch cluster operational (v{es_version}).",
                    version=es_version,
                    deployment_mode="Enterprise Search Engine (Lucene)"
                )
            else:
                return HealthCheckResult(
                    is_available=False,
                    status_message=f"Elasticsearch endpoint returned status {r.status_code}.",
                    deployment_mode="Enterprise Search Engine (Lucene)"
                )
        except Exception as e:
            return HealthCheckResult(
                is_available=False,
                status_message=f"Elasticsearch service not reachable on {self.host}. Requires running Elasticsearch Docker container.",
                deployment_mode="Enterprise Search Engine (Lucene)"
            )

    def connect(self) -> None:
        from elasticsearch import Elasticsearch
        self.client = Elasticsearch(self.host)
        self.is_connected = True

    def create_collection(self, collection_name: str, dimension: int, metric: str = "cosine") -> None:
        self.connect()
        self.index_name = collection_name
        self.dimension = dimension

        if self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)

        mappings = {
            "properties": {
                "doc_id": {"type": "keyword"},
                "vector": {
                    "type": "dense_vector",
                    "dims": dimension,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        }
        self.client.indices.create(index=self.index_name, mappings=mappings)

    def insert(self, vectors: np.ndarray, payloads: List[Dict[str, Any]], ids: List[str]) -> float:
        if self.client is None:
            raise RuntimeError("Elasticsearch client not connected.")

        t0 = time.perf_counter()
        from elasticsearch.helpers import bulk

        actions = [
            {
                "_index": self.index_name,
                "_id": str(did),
                "_source": {
                    "doc_id": str(did),
                    "vector": vec.tolist(),
                    **(payloads[idx] if payloads else {})
                }
            }
            for idx, (did, vec) in enumerate(zip(ids, vectors))
        ]
        bulk(self.client, actions, refresh=True)
        return time.perf_counter() - t0

    def build_index(self, index_params: Optional[Dict[str, Any]] = None) -> float:
        if self.client is None:
            raise RuntimeError("Elasticsearch client not connected.")
        t0 = time.perf_counter()
        self.client.indices.refresh(index=self.index_name)
        return time.perf_counter() - t0

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        if self.client is None:
            raise RuntimeError("Elasticsearch client not connected.")

        knn_query = {
            "field": "vector",
            "query_vector": query_vector.tolist(),
            "k": top_k,
            "num_candidates": max(top_k * 5, 50)
        }

        res = self.client.search(
            index=self.index_name,
            knn=knn_query,
            size=top_k,
            _source=["doc_id"]
        )

        results = []
        for hit in res["hits"]["hits"]:
            doc_id = str(hit["_source"].get("doc_id", hit["_id"]))
            score = float(hit["_score"])
            results.append(SearchResult(doc_id=doc_id, score=score))
        return results

    def get_storage_size(self) -> Optional[int]:
        if self.client is not None:
            try:
                stats = self.client.indices.stats(index=self.index_name)
                return int(stats["indices"][self.index_name]["total"]["store"]["size_in_bytes"])
            except Exception:
                return None
        return None

    def cleanup(self) -> None:
        if self.client is not None:
            try:
                if self.client.indices.exists(index=self.index_name):
                    self.client.indices.delete(index=self.index_name)
                self.client.close()
            except Exception:
                pass
            self.client = None
            self.is_connected = False
