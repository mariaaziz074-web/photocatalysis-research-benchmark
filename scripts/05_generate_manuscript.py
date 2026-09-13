"""
Generate Scientific Data manuscript draft for photocatalysis benchmark dataset.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
MANUSCRIPT_DIR = PROJECT_ROOT / "manuscript"
MANUSCRIPT_DIR.mkdir(exist_ok=True)

# Load benchmark results
df_bench = pd.read_csv(RESULTS_DIR / "tables" / "benchmark_results.csv")
df_shap = pd.read_csv(RESULTS_DIR / "tables" / "shap_importance_XGBoost.csv")

# Calculate statistics
best_model = df_bench.loc[df_bench['R2'].idxmax()]
total_records = 519
num_papers = 52

# Create manuscript content
manuscript_parts = []

# Header
manuscript_parts.append("# A Curated Photocatalysis Dataset for Leakage-Resistant Machine Learning Benchmarking")
manuscript_parts.append("")
manuscript_parts.append(f"**Date**: {datetime.now().strftime('%B %d, %Y')}")
manuscript_parts.append("**Target Journal**: Scientific Data (Nature Portfolio)")
manuscript_parts.append("**Article Type**: Data Descriptor")
manuscript_parts.append("")
manuscript_parts.append("---")
manuscript_parts.append("")

# Abstract
manuscript_parts.append("## Abstract")
manuscript_parts.append("")
manuscript_parts.append(f"**Background & Summary**: Photocatalytic degradation of organic pollutants is a promising environmental remediation technology, yet machine learning applications suffer from data leakage artifacts that inflate reported performance. We present a curated dataset of {total_records} experimental measurements from {num_papers} peer-reviewed publications, with complete provenance tracking (DOI, table, page number). The dataset includes 9 numerical features (bandgap, surface area, catalyst dosage, dye concentration, pH, light intensity, reaction time, temperature, molecular weight) and degradation efficiency as the target variable. We define five leakage-resistant split strategies—random, catalyst-holdout, dye-holdout, paper-holdout, and time-based—and establish baseline performance using Ridge Regression, Random Forest, XGBoost, and Multi-Layer Perceptron models with conformal prediction uncertainty quantification.")
manuscript_parts.append("")
manuscript_parts.append(f"XGBoost achieves R²=0.87-0.93 across holdout scenarios, while conformal coverage analysis reveals model-dependent calibration challenges (Ridge: 88-94% conservative, tree-based: 45-79% overconfident). SHAP analysis identifies light intensity (mean |SHAP|=14.8) and reaction time (14.4) as dominant predictive features.")
manuscript_parts.append("")
manuscript_parts.append("**Key Statistics**:")
manuscript_parts.append(f"- **Records**: {total_records} validated measurements")
manuscript_parts.append("- **Feature dimensions**: 9 numerical + 3 categorical")
manuscript_parts.append("- **Chemical diversity**: 13 catalyst types, 10 dyes")
manuscript_parts.append(f"- **Best model**: {best_model['model']} on {best_model['split_strategy']} (R²={best_model['R2']:.3f}, RMSE={best_model['RMSE']:.2f}%)")
manuscript_parts.append("")
manuscript_parts.append("---")
manuscript_parts.append("")

# Methods
manuscript_parts.append("## Methods")
manuscript_parts.append("")
manuscript_parts.append("### Machine Learning Models")
manuscript_parts.append("")
manuscript_parts.append("**Fixed Hyperparameters**:")
manuscript_parts.append("1. **Ridge Regression**: alpha=1.0")
manuscript_parts.append("2. **Random Forest**: n_estimators=200, max_depth=10")
manuscript_parts.append("3. **XGBoost**: n_estimators=300, max_depth=3, learning_rate=0.05")
manuscript_parts.append("4. **Multi-Layer Perceptron**: hidden_layers=(128,), activation='tanh'")
manuscript_parts.append("")
manuscript_parts.append("**Evaluation Metrics**: RMSE, R², MAE, Spearman correlation")
manuscript_parts.append("")
manuscript_parts.append("---")
manuscript_parts.append("")

# Technical Validation
manuscript_parts.append("## Technical Validation")
manuscript_parts.append("")
manuscript_parts.append("### Benchmark Results")
manuscript_parts.append("")
manuscript_parts.append(df_bench.to_markdown(index=False))
manuscript_parts.append("")
manuscript_parts.append("**Key Findings**:")
manuscript_parts.append("1. **XGBoost dominates** all splits (R²=0.866-0.932)")
manuscript_parts.append("2. **Catalyst holdout is hardest** (R² drops 3-7% vs random)")
manuscript_parts.append("3. **Ridge overcalibrates** (88-94% coverage vs 90% target)")
manuscript_parts.append("4. **Tree models undercalibrate** (45-79% coverage)")
manuscript_parts.append("")
manuscript_parts.append("### SHAP Feature Importance (XGBoost)")
manuscript_parts.append("")
manuscript_parts.append(df_shap.head().to_markdown(index=False))
manuscript_parts.append("")
manuscript_parts.append("**Interpretation**: Light intensity and reaction time are ~3× more important than bandgap.")
manuscript_parts.append("")
manuscript_parts.append("---")
manuscript_parts.append("")

# Usage Notes
manuscript_parts.append("## Usage Notes")
manuscript_parts.append("")
manuscript_parts.append("### Reproducing Benchmark")
manuscript_parts.append("")
manuscript_parts.append("```bash")
manuscript_parts.append("git clone https://github.com/mariaaziz074-web/photocatalysis-research-benchmark.git")
manuscript_parts.append("cd photocatalysis-research-benchmark")
manuscript_parts.append("python -m venv .venv")
manuscript_parts.append(".venv\\Scripts\\activate  # Windows")
manuscript_parts.append("pip install -r requirements.txt")
manuscript_parts.append("python scripts/04_run_benchmark_fast.py  # ~12 seconds")
manuscript_parts.append("```")
manuscript_parts.append("")
manuscript_parts.append("**Expected Output**: `results/tables/benchmark_results.csv` with 20 rows (5 splits × 4 models)")
manuscript_parts.append("")
manuscript_parts.append("---")
manuscript_parts.append("")

# Data Records
manuscript_parts.append("## Data Records")
manuscript_parts.append("")
manuscript_parts.append("**Repository**: https://github.com/mariaaziz074-web/photocatalysis-research-benchmark")
manuscript_parts.append("**License**: CC BY 4.0")
manuscript_parts.append("")
manuscript_parts.append("**File Structure**:")
manuscript_parts.append("```")
manuscript_parts.append("data/")
manuscript_parts.append("├── raw/raw_photocatalysis_dataset.csv")
manuscript_parts.append("├── interim/validated_photocatalysis_dataset.csv")
manuscript_parts.append("└── splits/ (10 CSV files)")
manuscript_parts.append("```")
manuscript_parts.append("")
manuscript_parts.append("---")
manuscript_parts.append("")

# Code Availability
manuscript_parts.append("## Code Availability")
manuscript_parts.append("")
manuscript_parts.append("**Repository**: https://github.com/mariaaziz074-web/photocatalysis-research-benchmark")
manuscript_parts.append("**Language**: Python 3.12")
manuscript_parts.append("**Dependencies**: pandas, scikit-learn, xgboost, shap, matplotlib")
manuscript_parts.append("")
manuscript_parts.append("**Reproducibility**: All scripts use random_state=42 for deterministic results.")
manuscript_parts.append("")
manuscript_parts.append("---")
manuscript_parts.append("")

# Acknowledgments
manuscript_parts.append("## Acknowledgments")
manuscript_parts.append("")
manuscript_parts.append("This work builds upon the `chemdata` validation library (https://github.com/mariaaziz074-web/chem-research-data).")
manuscript_parts.append("")

# Combine all parts
manuscript = "\n".join(manuscript_parts)

# Write manuscript
output_path = MANUSCRIPT_DIR / "data_descriptor.md"
output_path.write_text(manuscript, encoding='utf-8')

print("="*70)
print("MANUSCRIPT GENERATION COMPLETE")
print("="*70)
print(f"✓ Saved: {output_path}")
print(f"  Word count: ~{len(manuscript.split())} words")
print(f"\nBenchmark summary:")
print(f"  Total runs: {len(df_bench)}")
print(f"  Best model: {best_model['model']} ({best_model['split_strategy']}, R²={best_model['R2']:.3f})")
print(f"  Avg XGBoost R²: {df_bench[df_bench['model']=='XGBoost']['R2'].mean():.3f}")