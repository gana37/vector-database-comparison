"""
pgvector Adapter: Relational Database Extension (PostgreSQL).
Leverages PostgreSQL transactional ACID tables with IVFFlat / HNSW vector indexing.
"""
import time
import os
import socket
import numpy as np
from typing import Dict, List, Any, Optional
from src.adapters.base import BaseVectorDBAdapter, SearchResult, HealthCheckResult
from src.utils.logging_utils import setup_logger

logger = setup_logger("pgvector_adapter")

class PGVectorAdapter(BaseVectorDBAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__("pgvector", config)
        self.host = config.get("host", "localhost")
        self.port = int(config.get("port", 5432))
        self.database = config.get("database", "vectordb")
        self.user = config.get("user", "postgres")
        self.password_env = config.get("password_env", "PGPASSWORD")
        self.password = os.environ.get(self.password_env, "postgres")
        self.table_name = config.get("table_name", "scifact_vectors")
        self.conn = None

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
            import psycopg2
        except ImportError:
            return HealthCheckResult(
                is_available=False,
                status_message="psycopg2 package not installed.",
                deployment_mode="Relational Database Extension (PostgreSQL)"
            )

        if not self._is_port_open():
            return HealthCheckResult(
                is_available=False,
                status_message=f"PostgreSQL server not reachable on {self.host}:{self.port}. Requires running PostgreSQL instance with pgvector extension.",
                deployment_mode="Relational Database Extension (PostgreSQL)"
            )

        try:
            import psycopg2
            conn = psycopg2.connect(
                host=self.host, port=self.port, dbname=self.database,
                user=self.user, password=self.password, connect_timeout=2
            )
            cur = conn.cursor()
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
            row = cur.fetchone()
            cur.close()
            conn.close()
            if row:
                return HealthCheckResult(
                    is_available=True,
                    status_message=f"pgvector extension operational (v{row[0]}).",
                    version=row[0],
                    deployment_mode="Relational Database Extension (PostgreSQL)"
                )
            else:
                return HealthCheckResult(
                    is_available=False,
                    status_message="PostgreSQL is accessible, but 'vector' extension is not installed (run CREATE EXTENSION vector;).",
                    deployment_mode="Relational Database Extension (PostgreSQL)"
                )
        except Exception as e:
            return HealthCheckResult(
                is_available=False,
                status_message=f"PostgreSQL connection failed: {e}",
                deployment_mode="Relational Database Extension (PostgreSQL)"
            )

    def connect(self) -> None:
        import psycopg2
        self.conn = psycopg2.connect(
            host=self.host, port=self.port, dbname=self.database,
            user=self.user, password=self.password
        )
        self.conn.autocommit = True
        self.is_connected = True

    def create_collection(self, collection_name: str, dimension: int, metric: str = "cosine") -> None:
        self.connect()
        self.table_name = collection_name
        self.dimension = dimension
        
        cur = self.conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(f"DROP TABLE IF EXISTS {self.table_name};")
        cur.execute(f"""
            CREATE TABLE {self.table_name} (
                id SERIAL PRIMARY KEY,
                doc_id VARCHAR(64) NOT NULL,
                embedding vector({dimension}) NOT NULL
            );
        """)
        cur.close()

    def insert(self, vectors: np.ndarray, payloads: List[Dict[str, Any]], ids: List[str]) -> float:
        if self.conn is None:
            raise RuntimeError("Database not connected.")

        t0 = time.perf_counter()
        cur = self.conn.cursor()
        
        records = [
            (str(did), vec.tolist())
            for did, vec in zip(ids, vectors)
        ]
        from psycopg2.extras import execute_values
        execute_values(
            cur,
            f"INSERT INTO {self.table_name} (doc_id, embedding) VALUES %s",
            records,
            template="(%s, %s::vector)"
        )
        cur.close()
        return time.perf_counter() - t0

    def build_index(self, index_params: Optional[Dict[str, Any]] = None) -> float:
        if self.conn is None:
            raise RuntimeError("Database not connected.")

        t0 = time.perf_counter()
        cur = self.conn.cursor()
        cur.execute(f"""
            CREATE INDEX ON {self.table_name}
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 100);
        """)
        cur.close()
        return time.perf_counter() - t0

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        if self.conn is None:
            raise RuntimeError("Database not connected.")

        cur = self.conn.cursor()
        vec_str = str(query_vector.tolist())
        cur.execute(f"""
            SELECT doc_id, 1 - (embedding <=> %s::vector) AS score
            FROM {self.table_name}
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (vec_str, vec_str, top_k))

        rows = cur.fetchall()
        cur.close()
        return [SearchResult(doc_id=r[0], score=float(r[1])) for r in rows]

    def get_storage_size(self) -> Optional[int]:
        if self.conn is not None:
            try:
                cur = self.conn.cursor()
                cur.execute(f"SELECT pg_total_relation_size('{self.table_name}');")
                size = cur.fetchone()[0]
                cur.close()
                return int(size)
            except Exception:
                return None
        return None

    def cleanup(self) -> None:
        if self.conn is not None:
            try:
                cur = self.conn.cursor()
                cur.execute(f"DROP TABLE IF EXISTS {self.table_name};")
                cur.close()
                self.conn.close()
            except Exception:
                pass
            self.conn = None
            self.is_connected = False
