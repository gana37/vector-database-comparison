"""
Pinecone Adapter: Managed Cloud SaaS Vector Database.
Fully managed serverless/pod-based vector infrastructure.
Requires PINECONE_API_KEY environment variable for live benchmarking.
"""
import time
import os
import numpy as np
from typing import Dict, List, Any, Optional
from src.adapters.base import BaseVectorDBAdapter, SearchResult, HealthCheckResult
from src.utils.logging_utils import setup_logger

logger = setup_logger("pinecone_adapter")

class PineconeAdapter(BaseVectorDBAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Pinecone", config)
        self.api_key_env = config.get("api_key_env", "PINECONE_API_KEY")
        self.api_key = os.environ.get(self.api_key_env, "").strip()
        self.index_name = config.get("index_name", "scifact-benchmark")
        self.pc = None
        self.index = None

    def health_check(self) -> HealthCheckResult:
        if not self.api_key:
            return HealthCheckResult(
                is_available=False,
                status_message="API Credentials Required: PINECONE_API_KEY environment variable is not configured. Cloud SaaS live benchmark skipped to avoid unauthenticated access and prevent cloud billing charges.",
                deployment_mode="Managed Cloud SaaS"
            )

        try:
            from pinecone import Pinecone
            self.pc = Pinecone(api_key=self.api_key)
            indexes = self.pc.list_indexes()
            return HealthCheckResult(
                is_available=True,
                status_message="Pinecone cloud API authenticated successfully.",
                deployment_mode="Managed Cloud SaaS"
            )
        except ImportError:
            return HealthCheckResult(
                is_available=False,
                status_message="pinecone-client package not installed.",
                deployment_mode="Managed Cloud SaaS"
            )
        except Exception as e:
            return HealthCheckResult(
                is_available=False,
                status_message=f"Pinecone API authentication failed: {e}",
                deployment_mode="Managed Cloud SaaS"
            )

    def connect(self) -> None:
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required.")
        from pinecone import Pinecone
        self.pc = Pinecone(api_key=self.api_key)
        self.is_connected = True

    def create_collection(self, collection_name: str, dimension: int, metric: str = "cosine") -> None:
        self.connect()
        from pinecone import ServerlessSpec

        self.index_name = collection_name
        self.dimension = dimension

        existing = [idx.name for idx in self.pc.list_indexes()]
        if collection_name in existing:
            self.pc.delete_index(collection_name)

        self.pc.create_index(
            name=collection_name,
            dimension=dimension,
            metric="cosine" if metric.lower() == "cosine" else "dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        self.index = self.pc.Index(collection_name)

    def insert(self, vectors: np.ndarray, payloads: List[Dict[str, Any]], ids: List[str]) -> float:
        if self.index is None:
            raise RuntimeError("Pinecone index not initialized.")

        t0 = time.perf_counter()
        batch_size = 100
        total = len(ids)

        for i in range(0, total, batch_size):
            batch_vectors = vectors[i : i + batch_size].tolist()
            batch_ids = [str(did) for did in ids[i : i + batch_size]]
            records = [
                {"id": did, "values": vec, "metadata": {"doc_id": did}}
                for did, vec in zip(batch_ids, batch_vectors)
            ]
            self.index.upsert(vectors=records)

        return time.perf_counter() - t0

    def build_index(self, index_params: Optional[Dict[str, Any]] = None) -> float:
        # Pinecone indexes in background; allow brief stabilization
        time.sleep(2.0)
        return 2.0

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        if self.index is None:
            raise RuntimeError("Pinecone index not initialized.")

        res = self.index.query(
            vector=query_vector.tolist(),
            top_k=top_k,
            include_metadata=True
        )

        results = []
        for match in res.matches:
            doc_id = str(match.id)
            score = float(match.score)
            results.append(SearchResult(doc_id=doc_id, score=score, payload=dict(match.metadata or {})))
        return results

    def get_storage_size(self) -> Optional[int]:
        return None

    def cleanup(self) -> None:
        if self.pc is not None and self.index_name:
            try:
                existing = [idx.name for idx in self.pc.list_indexes()]
                if self.index_name in existing:
                    self.pc.delete_index(self.index_name)
            except Exception:
                pass
            self.index = None
            self.pc = None
            self.is_connected = False
