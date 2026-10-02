"""
Script to validate all 8 Vector Database Adapters individually.
Generates Phase 4 validation report.
"""
import os
import sys
import json
import yaml

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.benchmark.validator import DatabaseValidator
from src.utils.logging_utils import setup_logger

logger = setup_logger("validate_adapters")

def main():
    config_path = "config/benchmark_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("=" * 60)
    logger.info("PHASE 4: LOCAL DATABASE VALIDATION & SMOKE TESTING")
    logger.info("=" * 60)

    validator = DatabaseValidator(config)
    reports = validator.validate_all()

    # Save to disk
    out_dir = "outputs/logs"
    os.makedirs(out_dir, exist_ok=True)
    report_file = os.path.join(out_dir, "validation_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)

    print("\n" + "=" * 80)
    print("PHASE 4: ADAPTER VALIDATION REPORT (8 VECTOR DATABASES)")
    print("=" * 80)
    print(f"{'VDB Name':<15} | {'Deployment Model':<32} | {'Available':<10} | {'Smoke Test':<10}")
    print("-" * 80)
    for r in reports:
        avail_str = "YES" if r["is_available"] else "NO"
        smoke_str = "PASSED" if r["smoke_test_passed"] else "SKIPPED/FAIL"
        print(f"{r['name']:<15} | {r['deployment_mode']:<32} | {avail_str:<10} | {smoke_str:<10}")
        if not r["is_available"]:
            print(f"  -> Reason: {r['status_message']}")
    print("-" * 80)
    print(f"Validation report saved to: {report_file}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
