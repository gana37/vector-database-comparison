"""
Master CLI Entry Point for Vector Database Comparative Analysis and Benchmarking Platform.
Orchestrates data preparation, embedding generation, adapter validation, benchmarking, and Excel reporting.
"""
import os
import sys
import argparse
import yaml
import json

from src.reporting.excel_generator import generate_excel_report
from src.analysis.data_loader import export_structured_data
from src.utils.logging_utils import setup_logger

logger = setup_logger("main")

def load_config(config_path: str = "config/benchmark_config.yaml") -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_prepare_data(config: dict):
    logger.info("=" * 70)
    logger.info("STEP 1: PREPARING DATASET (BEIR/SciFact)")
    logger.info("=" * 70)
    from src.data.beir_loader import BEIRDataLoader
    loader = BEIRDataLoader(config)
    loader.download_dataset()
    corpus, queries, qrels = loader.load_raw_data()
    corpus_sub, queries_sub, qrels_sub = loader.create_deterministic_subset(corpus, queries, qrels)
    logger.info(f"Dataset ready: {len(corpus_sub)} docs, {len(queries_sub)} queries, {len(qrels_sub)} qrels.")

def run_generate_embeddings(config: dict):
    logger.info("=" * 70)
    logger.info("STEP 2: GENERATING EMBEDDINGS (all-MiniLM-L6-v2 ONNX)")
    logger.info("=" * 70)
    proc_dir = config.get("dataset", {}).get("processed_dir", "data/processed")
    corpus_path = os.path.join(proc_dir, "corpus_subset.json")
    queries_path = os.path.join(proc_dir, "queries_subset.json")
    if not os.path.exists(corpus_path) or not os.path.exists(queries_path):
        raise FileNotFoundError("Processed dataset files missing. Please run --prepare-data first.")

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    from src.embeddings.embedder import VectorEmbedder
    embedder = VectorEmbedder(config)
    result = embedder.generate_and_cache(corpus, queries)
    logger.info(f"Embeddings ready: Docs shape {result['doc_shape']}, Queries shape {result['query_shape']}")

def run_validate_adapters(config: dict):
    logger.info("=" * 70)
    logger.info("STEP 3: VALIDATING VECTOR DB ADAPTERS (SMOKE TESTS)")
    logger.info("=" * 70)
    from src.benchmark.validator import DatabaseValidator
    validator = DatabaseValidator(config)
    results = validator.validate_all()
    passed = sum(1 for r in results.values() if r.get("smoke_test_passed"))
    logger.info(f"Adapter validation completed: {passed}/{len(results)} adapters operational.")

def run_benchmark(config: dict):
    logger.info("=" * 70)
    logger.info("STEP 4: RUNNING MASTER BENCHMARK (8 VECTOR DATABASES)")
    logger.info("=" * 70)
    from src.benchmark.runner import BenchmarkRunner
    runner = BenchmarkRunner(config)
    results = runner.run_all()
    logger.info("Benchmark execution completed successfully.")
    return results

def run_generate_report():
    logger.info("=" * 70)
    logger.info("STEP 5: GENERATING EXCEL REPORT (VDB_Comprehensive_Analysis.xlsx)")
    logger.info("=" * 70)
    output_path = "outputs/excel/VDB_Comprehensive_Analysis.xlsx"
    generate_excel_report(output_path=output_path)
    logger.info(f"Primary deliverable generated at: {output_path}")
    export_structured_data()
    logger.info("Unified structured dataset synchronized: data/vdb_comparison.json")

def run_export_data():
    logger.info("=" * 70)
    logger.info("SYNCHRONIZING STRUCTURED DATASET (data/vdb_comparison.json)")
    logger.info("=" * 70)
    path = export_structured_data()
    logger.info(f"Structured dataset written to: {path}")

def run_dashboard():
    logger.info("=" * 70)
    logger.info("STARTING STREAMLIT ANALYTICS DASHBOARD")
    logger.info("=" * 70)
    import subprocess
    app_path = os.path.join("dashboard", "app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    subprocess.run(cmd)

def run_assistant(query=None):
    from src.assistant.vdb_assistant import VDBAssistant
    assistant = VDBAssistant()
    if query:
        res = assistant.ask(query)
        print(f"\n[QUERY]: {query}")
        print(f"[INTENT]: {res['intent']} | [EVIDENCE]: {res['evidence_type']}")
        print(res["answer"])
        return
    from scripts.interactive_assistant import main as cli_main
    cli_main()

def main():
    parser = argparse.ArgumentParser(
        description="Vector Database Comparative Analysis and Benchmarking Platform (AgentAnalytics.AI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --all                 # Execute end-to-end pipeline
  python main.py --run-benchmark       # Run benchmark across all 8 VDBs
  python main.py --generate-report     # Generate Excel deliverable
  python main.py --dashboard           # Launch interactive analytics dashboard
  python main.py --assistant           # Start interactive comparison assistant
  python main.py --validate            # Test VDB adapter connectivity
        """
    )
    parser.add_argument("--all", action="store_true", help="Execute complete pipeline (data -> embeddings -> validate -> benchmark -> report -> export)")
    parser.add_argument("--prepare-data", action="store_true", help="Download and extract BEIR/SciFact subset")
    parser.add_argument("--generate-embeddings", action="store_true", help="Generate all-MiniLM-L6-v2 384-d embeddings")
    parser.add_argument("--validate", action="store_true", help="Run smoke test health checks across all 8 VDB adapters")
    parser.add_argument("--run-benchmark", action="store_true", help="Execute performance and IR quality benchmark")
    parser.add_argument("--generate-report", action="store_true", help="Generate final VDB_Comprehensive_Analysis.xlsx")
    parser.add_argument("--export-data", action="store_true", help="Export unified structured data/vdb_comparison.json")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit analytics dashboard")
    parser.add_argument("--assistant", nargs="?", const="", help="Launch natural-language comparison assistant (or pass a single query in quotes)")
    parser.add_argument("--config", default="config/benchmark_config.yaml", help="Path to YAML configuration file")

    args = parser.parse_args()

    # If no flags are provided, show help and exit
    if not (args.all or args.prepare_data or args.generate_embeddings or args.validate or args.run_benchmark or args.generate_report or args.export_data or args.dashboard or args.assistant is not None):
        parser.print_help()
        sys.exit(0)

    config = load_config(args.config)

    if args.all:
        print("\n" + "=" * 80)
        print("LAUNCHING END-TO-END VDB BENCHMARKING AND ANALYSIS PIPELINE")
        print("=" * 80 + "\n")
        run_prepare_data(config)
        run_generate_embeddings(config)
        run_validate_adapters(config)
        run_benchmark(config)
        run_generate_report()
        run_export_data()
        print("\n" + "=" * 80)
        print("END-TO-END PIPELINE COMPLETED SUCCESSFULLY!")
        print("Excel Report: outputs/excel/VDB_Comprehensive_Analysis.xlsx")
        print("Structured Data: data/vdb_comparison.json")
        print("=" * 80 + "\n")
        return

    if args.prepare_data:
        run_prepare_data(config)
    if args.generate_embeddings:
        run_generate_embeddings(config)
    if args.validate:
        run_validate_adapters(config)
    if args.run_benchmark:
        run_benchmark(config)
    if args.generate_report:
        run_generate_report()
    if args.export_data:
        run_export_data()
    if args.dashboard:
        run_dashboard()
    if args.assistant is not None:
        run_assistant(query=args.assistant if args.assistant else None)

if __name__ == "__main__":
    main()
