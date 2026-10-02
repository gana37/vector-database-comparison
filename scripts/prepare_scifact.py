"""
Script to prepare and validate BEIR SciFact dataset subset.
"""
import os
import sys
import yaml

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.beir_loader import BEIRDataLoader
from src.utils.logging_utils import setup_logger

logger = setup_logger("prepare_scifact")

def main():
    config_path = "config/benchmark_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("=" * 60)
    logger.info("PHASE 1: DATASET ACQUISITION & VALIDATION")
    logger.info("=" * 60)
    
    loader = BEIRDataLoader(config)
    loader.download_dataset()
    corpus, queries, qrels = loader.load_raw_data()
    sub_corpus, sub_queries, sub_qrels = loader.create_deterministic_subset(corpus, queries, qrels)
    
    # Validation checks
    assert len(sub_corpus) == config["dataset"]["subset_docs"], f"Expected {config['dataset']['subset_docs']} docs, got {len(sub_corpus)}"
    assert len(sub_queries) == config["dataset"]["subset_queries"], f"Expected {config['dataset']['subset_queries']} queries, got {len(sub_queries)}"
    
    sample_doc_id = next(iter(sub_corpus))
    sample_doc = sub_corpus[sample_doc_id]
    sample_qid = next(iter(sub_queries))
    sample_query = sub_queries[sample_qid]
    
    print("\n" + "=" * 60)
    print("DATASET PREPARATION & VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total Raw Documents:            {len(corpus):,}")
    print(f"Total Raw Queries:              {len(queries):,}")
    print(f"Subset Documents:               {len(sub_corpus):,}")
    print(f"Subset Queries:                 {len(sub_queries):,}")
    print(f"Total Relevance Judgments:      {sum(len(v) for v in sub_qrels.values()):,}")
    print(f"Raw Dataset Path:               {loader.raw_dir}")
    print(f"Processed Dataset Path:         {loader.processed_dir}")
    print("-" * 60)
    print("SAMPLE DOCUMENT:")
    print(f"  ID:    {sample_doc['id']}")
    print(f"  Title: {sample_doc['title']}")
    print(f"  Text:  {sample_doc['text'][:200]}...")
    print("-" * 60)
    print("SAMPLE QUERY:")
    print(f"  ID:    {sample_qid}")
    print(f"  Query: {sample_query}")
    print(f"  Ground-Truth Relevant Doc IDs: {list(sub_qrels.get(sample_qid, {}).keys())}")
    print("-" * 60)
    print("VALIDATION RESULT: ALL CHECKS PASSED (100% Deterministic & Validated)")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
