"""
Isolated smoke-testing and validation harness for all 8 Vector Database Adapters.
Runs a 5-vector verification cycle before entering production benchmarking.
"""
import time
import numpy as np
from typing import Dict, Any, List
from src.adapters import ADAPTER_REGISTRY, get_adapter, HealthCheckResult
from src.utils.logging_utils import setup_logger

logger = setup_logger("validator")

class DatabaseValidator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def validate_single_adapter(self, db_key: str) -> Dict[str, Any]:
        """Runs a complete connection, insert, query, and cleanup cycle on 1 adapter."""
        logger.info(f"--- Validating Adapter: {db_key.upper()} ---")
        adapter = get_adapter(db_key, self.config)
        
        # 1. Health check probe
        health: HealthCheckResult = adapter.health_check()
        result = {
            "db_key": db_key,
            "name": adapter.name,
            "deployment_mode": health.deployment_mode,
            "version": health.version,
            "is_available": health.is_available,
            "status_message": health.status_message,
            "smoke_test_passed": False,
            "error": None
        }

        if not health.is_available:
            logger.warning(f"[{adapter.name}] Not available for live benchmarking: {health.status_message}")
            return result

        # 2. Smoke test: 5 vectors of dim 384
        try:
            test_dim = 384
            np.random.seed(42)
            test_vectors = np.random.randn(5, test_dim).astype(np.float32)
            norms = np.linalg.norm(test_vectors, axis=1, keepdims=True)
            test_vectors = test_vectors / norms  # unit normalize
            
            test_ids = ["test_doc_0", "test_doc_1", "test_doc_2", "test_doc_3", "test_doc_4"]
            test_payloads = [{"title": f"Doc {i}", "category": "test"} for i in range(5)]

            test_collection = f"test_{db_key}_smoke"
            logger.info(f"[{adapter.name}] Creating collection '{test_collection}'...")
            adapter.create_collection(test_collection, dimension=test_dim, metric="cosine")

            logger.info(f"[{adapter.name}] Inserting 5 test vectors...")
            insert_time = adapter.insert(test_vectors, test_payloads, test_ids)

            logger.info(f"[{adapter.name}] Building index...")
            adapter.build_index()

            logger.info(f"[{adapter.name}] Searching with test query...")
            query_vec = test_vectors[0]
            search_res = adapter.search(query_vec, top_k=3)

            assert len(search_res) > 0, "No results returned from smoke test query!"
            top_hit = search_res[0]
            logger.info(f"[{adapter.name}] Top hit: ID={top_hit.doc_id}, score={top_hit.score:.4f}")
            assert top_hit.doc_id == "test_doc_0", f"Expected top hit 'test_doc_0', got '{top_hit.doc_id}'"

            logger.info(f"[{adapter.name}] Cleaning up test collection...")
            adapter.cleanup()

            result["smoke_test_passed"] = True
            result["status_message"] = "Verified: Connect, Insert, Search, and Cleanup cycle succeeded."
            logger.info(f"[{adapter.name}] SMOKE TEST PASSED SUCCESSFULLY.")

        except Exception as e:
            logger.error(f"[{adapter.name}] Smoke test failed: {e}")
            result["smoke_test_passed"] = False
            result["error"] = str(e)
            try:
                adapter.cleanup()
            except Exception:
                pass

        return result

    def validate_all(self) -> List[Dict[str, Any]]:
        """Validates all 8 vector database adapters."""
        results = []
        for db_key in ADAPTER_REGISTRY.keys():
            res = self.validate_single_adapter(db_key)
            results.append(res)
        return results
