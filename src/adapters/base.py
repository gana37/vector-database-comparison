"""
Abstract Base Adapter for Vector Database Benchmarking.
Defines the uniform lifecycle and query contract for all 8 vector databases.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class SearchResult:
    doc_id: str
    score: float
    payload: Optional[Dict[str, Any]] = None

@dataclass
class HealthCheckResult:
    is_available: bool
    status_message: str
    version: Optional[str] = None
    deployment_mode: str = "Unknown"

class BaseVectorDBAdapter(ABC):
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.collection_name = config.get("collection_name", "scifact_benchmark")
        self.dimension = config.get("dimension", 384)
        self.metric = config.get("metric", "cosine")
        self.is_connected = False

    @abstractmethod
    def health_check(self) -> HealthCheckResult:
        """
        Tests whether the database daemon, embedded library, or cloud API is genuinely accessible.
        Returns HealthCheckResult indicating availability and reason if not available.
        """
        pass

    @abstractmethod
    def connect(self) -> None:
        """Establishes connection or initializes local database client."""
        pass

    @abstractmethod
    def create_collection(self, collection_name: str, dimension: int, metric: str = "cosine") -> None:
        """Creates or resets an index/collection with the specified dimension and similarity metric."""
        pass

    @abstractmethod
    def insert(self, vectors: np.ndarray, payloads: List[Dict[str, Any]], ids: List[str]) -> float:
        """
        Inserts a batch of vectors with corresponding payloads and IDs.
        Returns wall-clock insertion duration in seconds.
        """
        pass

    @abstractmethod
    def build_index(self, index_params: Optional[Dict[str, Any]] = None) -> float:
        """
        Explicitly triggers index construction or optimization if required by the architecture.
        Returns wall-clock index build duration in seconds.
        """
        pass

    def warmup(self, query_vectors: np.ndarray, top_k: int = 10) -> None:
        """Executes warm-up queries to stabilize internal caches and socket connections."""
        for i in range(len(query_vectors)):
            self.search(query_vectors[i], top_k=top_k)

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Executes Top-K vector similarity search.
        Returns a list of SearchResult objects sorted by relevance score descending.
        """
        pass

    @abstractmethod
    def get_storage_size(self) -> Optional[int]:
        """Returns on-disk storage size in bytes if measurable, or None."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Deletes collection and frees local or remote resources."""
        pass
