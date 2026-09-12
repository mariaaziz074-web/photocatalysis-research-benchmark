"""
scripts/03_run_validation_and_eda.py
===================================
Master orchestration script to execute data validation, quality assessment, and EDA figure generation.
"""

import os
import sys
import logging
from pathlib import Path

# Ensure src is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from photocatalysis_benchmark.validation.pipeline import DatasetValidationPipeline
from photocatalysis_benchmark.validation.quality_report import generate_quality_report
from photocatalysis_benchmark.analysis.eda import generate_all_eda_visualizations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("validation_eda_runner")


def main():
    logger.info("================================================================")
    logger.info("   PHOTOCATALYSIS BENCHMARK: VALIDATION & EDA PIPELINE")
    logger.info("================================================================")

    raw_csv = PROJECT_ROOT / "data" / "raw" / "raw_photocatalysis_dataset.csv"
    interim_csv = PROJECT_ROOT / "data" / "interim" / "validated_photocatalysis_dataset.csv"
    report_md = PROJECT_ROOT / "results" / "reports" / "data_quality_report.md"
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"

    if not raw_csv.exists():
        logger.error(f"Raw data file not found at {raw_csv}. Please run 'python scripts/02_import_raw_data.py' first.")
        sys.exit(1)

    # 1. Execute Validation & Cleaning Pipeline
    logger.info("--> Executing Chemical & Numerical Validation Pipeline...")
    pipeline = DatasetValidationPipeline()
    df_validated = pipeline.run(input_csv=raw_csv, output_csv=interim_csv)

    # 2. Generate Technical Validation Quality Report
    logger.info("--> Generating FAIR Data Quality Report and Summary Tables...")
    generate_quality_report(df=df_validated, output_md_path=report_md, output_table_dir=tables_dir)

    # 3. Generate Scientific Figures
    logger.info("--> Generating Publication-Quality Figures (Figs 1-4)...")
    generate_all_eda_visualizations(df=df_validated, figures_dir=figures_dir)

    logger.info("================================================================")
    logger.info(" Pipeline execution completed successfully!")
    logger.info(f" Validated Dataset: {interim_csv}")
    logger.info(f" Quality Report:    {report_md}")
    logger.info(f" Summary Tables:    {tables_dir}")
    logger.info(f" Figures (300 DPI): {figures_dir}")
    logger.info("================================================================")


if __name__ == "__main__":
    main()