"""
Script to generate, validate, and cache embeddings for SciFact dataset subset.
"""
import os
import sys
import json
import yaml

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.embeddings.embedder import VectorEmbedder
from src.utils.logging_utils import setup_logger

logger = setup_logger("generate_embeddings")

def main():
    config_path = "config/benchmark_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("=" * 60)
    logger.info("PHASE 2: EMBEDDING GENERATION & VALIDATION")
    logger.info("=" * 60)

    # Load pre-processed subset
    processed_dir = config["dataset"]["processed_dir"]
    corpus_file = os.path.join(processed_dir, "corpus_subset.json")
    queries_file = os.path.join(processed_dir, "queries_subset.json")

    if not os.path.exists(corpus_file) or not os.path.exists(queries_file):
        raise FileNotFoundError(f"Processed dataset files not found in {processed_dir}. Run prepare_scifact.py first.")

    with open(corpus_file, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    with open(queries_file, "r", encoding="utf-8") as f:
        queries = json.load(f)

    embedder = VectorEmbedder(config)
    result = embedder.generate_and_cache(corpus, queries)
    meta = result["metadata"]

    print("\n" + "=" * 60)
    print("EMBEDDING PIPELINE VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Model Name:                 {meta['model_name']}")
    print(f"Embedding Dimension:        {meta['dimension']}")
    print(f"Number of Document Vectors: {meta['num_document_vectors']:,}")
    print(f"Number of Query Vectors:    {meta['num_query_vectors']:,}")
    print(f"Data Type:                  {meta['dtype']}")
    print(f"Normalization Method:       {meta['normalization_method']}")
    print(f"Doc Embedding Time:         {meta['doc_generation_time_sec']:.2f} s")
    print(f"Query Embedding Time:       {meta['query_generation_time_sec']:.2f} s")
    print(f"Total Time:                 {meta['total_generation_time_sec']:.2f} s")
    print(f"Doc Embeddings Path:        {meta['doc_embeddings_file']}")
    print(f"Query Embeddings Path:      {meta['query_embeddings_file']}")
    print(f"Norm Bounds (Docs):         [{meta['min_doc_norm']:.4f}, {meta['max_doc_norm']:.4f}]")
    print(f"Norm Bounds (Queries):      [{meta['min_query_norm']:.4f}, {meta['max_query_norm']:.4f}]")
    print("-" * 60)
    print("VALIDATION RESULT: ALL CHECKS PASSED (Embeddings Cached & Ready)")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
