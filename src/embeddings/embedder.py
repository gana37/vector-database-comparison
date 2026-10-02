"""
Embedding generation pipeline using official Sentence Transformers (all-MiniLM-L6-v2) weights.
Uses ONNXRuntime + HuggingFace Tokenizers for high-speed, portable, zero-lock inference on Windows / Python 3.14.
Generates unit-normalized 384-dimensional dense vectors once and caches them to disk.
"""
import os
import json
import time
import requests
import numpy as np
import onnxruntime as ort
from typing import Dict, List, Tuple, Any
from tokenizers import Tokenizer
from src.utils.logging_utils import setup_logger

logger = setup_logger("embedder")

class VectorEmbedder:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embed_cfg = config.get("embedding", {})
        self.model_name = self.embed_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        self.dimension = self.embed_cfg.get("dimension", 384)
        self.normalize_l2 = self.embed_cfg.get("normalize_l2", True)
        self.batch_size = self.embed_cfg.get("batch_size", 64)
        self.cache_dir = self.embed_cfg.get("cache_dir", "data/embeddings")
        self.models_dir = os.path.join("data", "models", "all-MiniLM-L6-v2")
        
        self.tokenizer = None
        self.session = None

    def _ensure_model_files(self) -> Tuple[str, str]:
        """Downloads official HuggingFace ONNX weights and tokenizer if not present."""
        os.makedirs(self.models_dir, exist_ok=True)
        tok_path = os.path.join(self.models_dir, "tokenizer.json")
        onnx_path = os.path.join(self.models_dir, "model.onnx")
        
        headers = {"User-Agent": "Mozilla/5.0"}
        
        if not os.path.exists(tok_path) or os.path.getsize(tok_path) < 1000:
            logger.info("Downloading all-MiniLM-L6-v2 tokenizer.json...")
            r = requests.get(
                "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/raw/main/tokenizer.json",
                headers=headers,
                timeout=60
            )
            r.raise_for_status()
            with open(tok_path, "wb") as f:
                f.write(r.content)
                
        if not os.path.exists(onnx_path) or os.path.getsize(onnx_path) < 10000000:
            logger.info("Downloading all-MiniLM-L6-v2 model.onnx (~90MB)...")
            r = requests.get(
                "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx",
                headers=headers,
                stream=True,
                timeout=120
            )
            r.raise_for_status()
            with open(onnx_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        
        return tok_path, onnx_path

    def load_model(self):
        """Initializes fast Tokenizer and ONNXRuntime InferenceSession."""
        if self.session is None:
            tok_path, onnx_path = self._ensure_model_files()
            logger.info(f"Loading Tokenizer from {tok_path}...")
            self.tokenizer = Tokenizer.from_file(tok_path)
            self.tokenizer.enable_padding(pad_token="[PAD]", pad_id=0, length=256)
            self.tokenizer.enable_truncation(max_length=256)
            
            logger.info(f"Loading ONNXRuntime InferenceSession from {onnx_path}...")
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
            self.session = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
            logger.info("Loaded all-MiniLM-L6-v2 ONNX engine successfully.")

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encodes texts in batches and applies mean pooling + L2 normalization."""
        self.load_model()
        all_embeddings = []
        
        total = len(texts)
        for i in range(0, total, self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            encoded_batch = self.tokenizer.encode_batch(batch_texts)
            
            input_ids = np.array([e.ids for e in encoded_batch], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encoded_batch], dtype=np.int64)
            token_type_ids = np.array([e.type_ids for e in encoded_batch], dtype=np.int64)
            
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids
            }
            
            outputs = self.session.run(None, inputs)
            last_hidden_state = outputs[0]  # (B, seq_len, 384)
            
            # Mean pooling with attention mask
            input_mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
            sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, axis=1)
            sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
            mean_pooled = sum_embeddings / sum_mask
            
            # L2 Unit Normalization
            if self.normalize_l2:
                norm = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
                normalized = mean_pooled / np.clip(norm, a_min=1e-12, a_max=None)
            else:
                normalized = mean_pooled
                
            all_embeddings.append(normalized.astype(np.float32))
            
        return np.vstack(all_embeddings)

    def generate_and_cache(
        self,
        corpus: Dict[str, Dict[str, Any]],
        queries: Dict[str, str]
    ) -> Dict[str, Any]:
        """Generates document and query embeddings once and caches to disk."""
        os.makedirs(self.cache_dir, exist_ok=True)
        
        doc_emb_path = os.path.join(self.cache_dir, "doc_embeddings_384d.npy")
        query_emb_path = os.path.join(self.cache_dir, "query_embeddings_384d.npy")
        doc_ids_path = os.path.join(self.cache_dir, "doc_ids.json")
        query_ids_path = os.path.join(self.cache_dir, "query_ids.json")
        meta_path = os.path.join(self.cache_dir, "embedding_metadata.json")

        if (
            os.path.exists(doc_emb_path)
            and os.path.exists(query_emb_path)
            and os.path.exists(doc_ids_path)
            and os.path.exists(query_ids_path)
        ):
            logger.info(f"Embeddings already exist in {self.cache_dir}. Loading from cache...")
            doc_embeddings = np.load(doc_emb_path)
            query_embeddings = np.load(query_emb_path)
            with open(doc_ids_path, "r", encoding="utf-8") as f:
                doc_ids = json.load(f)
            with open(query_ids_path, "r", encoding="utf-8") as f:
                query_ids = json.load(f)
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return {
                "doc_embeddings": doc_embeddings,
                "query_embeddings": query_embeddings,
                "doc_ids": doc_ids,
                "query_ids": query_ids,
                "metadata": meta
            }

        doc_ids = list(corpus.keys())
        doc_texts = []
        for did in doc_ids:
            doc = corpus[did]
            title = doc.get("title", "").strip()
            text = doc.get("text", "").strip()
            full_text = f"{title} {text}".strip() if title else text
            doc_texts.append(full_text)

        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]

        logger.info(f"Generating embeddings for {len(doc_texts)} documents (dim={self.dimension})...")
        t0 = time.time()
        doc_embeddings = self.encode_texts(doc_texts)
        doc_time = time.time() - t0

        logger.info(f"Generating embeddings for {len(query_texts)} queries (dim={self.dimension})...")
        t1 = time.time()
        query_embeddings = self.encode_texts(query_texts)
        query_time = time.time() - t1

        # Validation Checks
        assert doc_embeddings.shape == (len(doc_ids), self.dimension), f"Doc shape mismatch: {doc_embeddings.shape}"
        assert query_embeddings.shape == (len(query_ids), self.dimension), f"Query shape mismatch: {query_embeddings.shape}"
        assert not np.isnan(doc_embeddings).any(), "Doc embeddings contain NaN values!"
        assert not np.isnan(query_embeddings).any(), "Query embeddings contain NaN values!"

        # Norm verification (Unit length L2 norm ≈ 1.0)
        doc_norms = np.linalg.norm(doc_embeddings, axis=1)
        query_norms = np.linalg.norm(query_embeddings, axis=1)
        assert np.allclose(doc_norms, 1.0, atol=1e-3), "Doc embeddings are not properly L2 normalized!"
        assert np.allclose(query_norms, 1.0, atol=1e-3), "Query embeddings are not properly L2 normalized!"

        # Save to disk
        np.save(doc_emb_path, doc_embeddings)
        np.save(query_emb_path, query_embeddings)
        with open(doc_ids_path, "w", encoding="utf-8") as f:
            json.dump(doc_ids, f, indent=2)
        with open(query_ids_path, "w", encoding="utf-8") as f:
            json.dump(query_ids, f, indent=2)

        meta = {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "num_document_vectors": len(doc_ids),
            "num_query_vectors": len(query_ids),
            "normalization_method": "L2 unit normalization (||v||=1.0)",
            "dtype": str(doc_embeddings.dtype),
            "doc_generation_time_sec": round(doc_time, 3),
            "query_generation_time_sec": round(query_time, 3),
            "total_generation_time_sec": round(doc_time + query_time, 3),
            "doc_embeddings_file": doc_emb_path,
            "query_embeddings_file": query_emb_path,
            "min_doc_norm": float(np.min(doc_norms)),
            "max_doc_norm": float(np.max(doc_norms)),
            "min_query_norm": float(np.min(query_norms)),
            "max_query_norm": float(np.max(query_norms))
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Embeddings successfully cached to {self.cache_dir}:")
        logger.info(f"  - Docs: {doc_embeddings.shape} in {doc_time:.2f}s")
        logger.info(f"  - Queries: {query_embeddings.shape} in {query_time:.2f}s")

        return {
            "doc_embeddings": doc_embeddings,
            "query_embeddings": query_embeddings,
            "doc_ids": doc_ids,
            "query_ids": query_ids,
            "metadata": meta
        }
