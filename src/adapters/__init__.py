"""
Vector Database Adapters Module.
Registers and exposes adapters for all 8 vector database systems.
"""
from typing import Dict, Any, Type
from src.adapters.base import BaseVectorDBAdapter, SearchResult, HealthCheckResult
from src.adapters.qdrant_adapter import QdrantAdapter
from src.adapters.chroma_adapter import ChromaAdapter
from src.adapters.faiss_adapter import FAISSAdapter
from src.adapters.milvus_adapter import MilvusAdapter
from src.adapters.weaviate_adapter import WeaviateAdapter
from src.adapters.pgvector_adapter import PGVectorAdapter
from src.adapters.elasticsearch_adapter import ElasticsearchAdapter
from src.adapters.pinecone_adapter import PineconeAdapter

ADAPTER_REGISTRY: Dict[str, Type[BaseVectorDBAdapter]] = {
    "qdrant": QdrantAdapter,
    "chroma": ChromaAdapter,
    "faiss": FAISSAdapter,
    "milvus": MilvusAdapter,
    "weaviate": WeaviateAdapter,
    "pgvector": PGVectorAdapter,
    "elasticsearch": ElasticsearchAdapter,
    "pinecone": PineconeAdapter,
}

def get_adapter(db_key: str, config: Dict[str, Any]) -> BaseVectorDBAdapter:
    """Instantiates a vector database adapter by key."""
    key = db_key.lower()
    if key not in ADAPTER_REGISTRY:
        raise ValueError(f"Unknown database key: {db_key}. Available: {list(ADAPTER_REGISTRY.keys())}")
    db_config = config.get("databases", {}).get(key, {})
    return ADAPTER_REGISTRY[key](db_config)

__all__ = [
    "BaseVectorDBAdapter",
    "SearchResult",
    "HealthCheckResult",
    "QdrantAdapter",
    "ChromaAdapter",
    "FAISSAdapter",
    "MilvusAdapter",
    "WeaviateAdapter",
    "PGVectorAdapter",
    "ElasticsearchAdapter",
    "PineconeAdapter",
    "ADAPTER_REGISTRY",
    "get_adapter",
]
