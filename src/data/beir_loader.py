"""
BEIR SciFact dataset loader, validator, and deterministic subsampler.
"""
import os
import json
import zipfile
import requests
from typing import Dict, List, Tuple, Any
from src.utils.logging_utils import setup_logger

logger = setup_logger("beir_loader")

class BEIRDataLoader:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dataset_cfg = config.get("dataset", {})
        self.raw_dir = self.dataset_cfg.get("raw_dir", "data/raw/scifact")
        self.processed_dir = self.dataset_cfg.get("processed_dir", "data/processed")
        self.url = self.dataset_cfg.get("url", "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip")
        self.subset_docs = self.dataset_cfg.get("subset_docs", 1000)
        self.subset_queries = self.dataset_cfg.get("subset_queries", 50)
        self.seed = self.dataset_cfg.get("seed", 42)

    def download_dataset(self) -> str:
        """Downloads the SciFact archive if not already present."""
        os.makedirs(os.path.dirname(self.raw_dir), exist_ok=True)
        zip_path = os.path.join(os.path.dirname(self.raw_dir), "scifact.zip")
        
        # Check if already extracted
        corpus_path = os.path.join(self.raw_dir, "corpus.jsonl")
        if os.path.exists(corpus_path):
            logger.info(f"SciFact dataset already downloaded and extracted at: {self.raw_dir}")
            return self.raw_dir
            
        if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 100000:
            logger.info(f"Downloading SciFact dataset from {self.url}...")
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    response = requests.get(self.url, headers=headers, stream=True, timeout=90)
                    response.raise_for_status()
                    total_downloaded = 0
                    with open(zip_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=128 * 1024):
                            if chunk:
                                f.write(chunk)
                                total_downloaded += len(chunk)
                    logger.info(f"Downloaded SciFact archive ({total_downloaded / 1024:.1f} KB)")
                    break
                except Exception as e:
                    logger.warning(f"Download attempt {attempt} failed: {e}")
                    if attempt == max_retries:
                        raise
            
        # Extract archive
        logger.info(f"Extracting {zip_path} to {os.path.dirname(self.raw_dir)}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(os.path.dirname(self.raw_dir))
            
        logger.info(f"Extracted dataset successfully to {self.raw_dir}")
        return self.raw_dir

    def load_raw_data(self) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str], Dict[str, Dict[str, int]]]:
        """Loads and parses raw corpus.jsonl, queries.jsonl, and qrels/test.tsv."""
        corpus_path = os.path.join(self.raw_dir, "corpus.jsonl")
        queries_path = os.path.join(self.raw_dir, "queries.jsonl")
        qrels_path = os.path.join(self.raw_dir, "qrels", "test.tsv")
        
        if not os.path.exists(corpus_path) or not os.path.exists(queries_path) or not os.path.exists(qrels_path):
            raise FileNotFoundError(f"Missing raw files in {self.raw_dir}. Ensure download_dataset() was called.")
            
        # 1. Load Corpus
        corpus = {}
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                doc_id = str(item["_id"])
                corpus[doc_id] = {
                    "id": doc_id,
                    "title": item.get("title", ""),
                    "text": item.get("text", ""),
                    "metadata": item.get("metadata", {})
                }
                
        # 2. Load Queries
        queries = {}
        with open(queries_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                queries[str(item["_id"])] = item.get("text", "")
                
        # 3. Load Qrels (test.tsv: query_id, corpus_id, score)
        qrels: Dict[str, Dict[str, int]] = {}
        with open(qrels_path, "r", encoding="utf-8") as f:
            header = f.readline()  # skip header: query-id\tcorpus-id\tscore
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    qid, doc_id, score = parts[0], parts[1], int(parts[2])
                    if qid not in qrels:
                        qrels[qid] = {}
                    qrels[qid][doc_id] = score

        logger.info(f"Parsed Raw SciFact Dataset: {len(corpus):,} docs, {len(queries):,} queries, {sum(len(v) for v in qrels.values()):,} qrel judgements")
        return corpus, queries, qrels

    def create_deterministic_subset(
        self,
        corpus: Dict[str, Dict[str, Any]],
        queries: Dict[str, str],
        qrels: Dict[str, Dict[str, int]]
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str], Dict[str, Dict[str, int]]]:
        """
        Creates a deterministic subset of 1,000 documents and 50 queries.
        CRITICAL: Ensures selected queries have at least one ground-truth document inside the 1,000-document set!
        """
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Select 50 queries that have qrels
        candidate_qids = [qid for qid, rels in qrels.items() if len(rels) > 0 and qid in queries]
        candidate_qids.sort() # deterministic sort
        
        selected_qids = candidate_qids[:self.subset_queries]
        selected_queries = {qid: queries[qid] for qid in selected_qids}
        
        # Gather all relevant document IDs for these selected queries
        must_include_doc_ids = set()
        for qid in selected_qids:
            for doc_id, score in qrels[qid].items():
                if score > 0 and doc_id in corpus:
                    must_include_doc_ids.add(doc_id)
                    
        # Fill remaining documents deterministically up to subset_docs (1,000)
        sorted_all_doc_ids = sorted(list(corpus.keys()))
        selected_doc_ids = set(must_include_doc_ids)
        
        for doc_id in sorted_all_doc_ids:
            if len(selected_doc_ids) >= self.subset_docs:
                break
            selected_doc_ids.add(doc_id)
            
        selected_corpus = {doc_id: corpus[doc_id] for doc_id in sorted(list(selected_doc_ids))}
        
        # Filter qrels to only include documents in the selected corpus
        selected_qrels: Dict[str, Dict[str, int]] = {}
        total_judgments = 0
        for qid in selected_qids:
            sub_rels = {doc_id: score for doc_id, score in qrels[qid].items() if doc_id in selected_corpus}
            selected_qrels[qid] = sub_rels
            total_judgments += len(sub_rels)
            
        # Serialize processed files
        corpus_out = os.path.join(self.processed_dir, "corpus_subset.json")
        queries_out = os.path.join(self.processed_dir, "queries_subset.json")
        qrels_out = os.path.join(self.processed_dir, "qrels_subset.json")
        meta_out = os.path.join(self.processed_dir, "subset_metadata.json")
        
        with open(corpus_out, "w", encoding="utf-8") as f:
            json.dump(selected_corpus, f, indent=2)
            
        with open(queries_out, "w", encoding="utf-8") as f:
            json.dump(selected_queries, f, indent=2)
            
        with open(qrels_out, "w", encoding="utf-8") as f:
            json.dump(selected_qrels, f, indent=2)
            
        metadata = {
            "dataset_name": "BEIR/SciFact",
            "total_raw_documents": len(corpus),
            "total_raw_queries": len(queries),
            "subset_documents": len(selected_corpus),
            "subset_queries": len(selected_queries),
            "subset_relevance_judgments": total_judgments,
            "queries_with_ground_truth": sum(1 for qid, r in selected_qrels.items() if len(r) > 0),
            "seed": self.seed
        }
        with open(meta_out, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Deterministic Subset Created Successfully:")
        logger.info(f"  - Documents: {len(selected_corpus):,}")
        logger.info(f"  - Queries: {len(selected_queries):,}")
        logger.info(f"  - Relevant Judgments in Subset: {total_judgments:,}")
        logger.info(f"  - Files saved to: {self.processed_dir}")
        
        return selected_corpus, selected_queries, selected_qrels
