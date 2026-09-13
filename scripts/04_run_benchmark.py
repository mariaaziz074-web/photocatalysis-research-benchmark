"""
scripts/04_run_benchmark.py
===========================
Complete benchmarking pipeline: 5 splits × 4 models = 20 experiments.
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from photocatalysis_benchmark.benchmarks.splitting import DataSplitter
from photocatalysis_benchmark.benchmarks.models import PhotocatalysisModelBenchmark
from photocatalysis_benchmark.benchmarks.conformal_prediction import ConformalPredictor
from photocatalysis_benchmark.benchmarks.explainability import SHAPExplainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("benchmark_runner")


def main():
    logger.info("="*80)
    logger.info("  PHOTOCATALYSIS BENCHMARK: ML MODEL EVALUATION PIPELINE")
    logger.info("="*80)

    # Paths
    data_path = PROJECT_ROOT / "data" / "interim" / "validated_photocatalysis_dataset.csv"
    results_dir = PROJECT_ROOT / "results"
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    models_dir = results_dir / "models"

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading validated dataset from {data_path}...")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} records")

    # Define features
    feature_cols = [
        "bandgap_ev",
        "surface_area_m2g",
        "catalyst_dosage_gl",
        "initial_dye_conc_mgl",
        "ph",
        "light_intensity_mwcm2",
        "reaction_time_min",
        "temperature_c",
        "dye_molecular_weight"
    ]

    target_col = "degradation_efficiency_percent"

    # Initialize splitter
    splitter = DataSplitter(df, test_size=0.2, random_state=42)
    splits = splitter.get_all_splits()

    # Results storage
    all_results = []

    # Run benchmark for each split strategy
    for split_name, (train_df, test_df) in splits.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"  SPLIT STRATEGY: {split_name.upper()}")
        logger.info(f"{'='*80}")

        # Initialize model benchmark
        benchmark = PhotocatalysisModelBenchmark(feature_cols, target_col)

        # Prepare data
        X_train, X_test, y_train, y_test = benchmark.prepare_data(train_df, test_df)

        # Train all models
        models = benchmark.train_all_models(X_train, y_train)

        # Evaluate each model
        for model_name, model in models.items():
            logger.info(f"\n--- Evaluating {model_name} on {split_name} ---")

            # Standard evaluation
            metrics = benchmark.evaluate_model(model, X_test, y_test)
            metrics['split_strategy'] = split_name
            metrics['model'] = model_name

            # Conformal prediction (using 20% of training as calibration)
            cal_size = int(0.2 * len(X_train))
            X_cal, y_cal = X_train[-cal_size:], y_train[-cal_size:]
            X_train_reduced = X_train[:-cal_size]
            y_train_reduced = y_train[:-cal_size]

            cp = ConformalPredictor(model, confidence_level=0.90)
            cp.calibrate(X_cal, y_cal)
            y_pred, lower, upper = cp.predict_with_interval(X_test)
            coverage = cp.compute_coverage(y_test, lower, upper)

            metrics['conformal_coverage'] = coverage
            metrics['conformal_interval_width'] = np.mean(upper - lower)

            all_results.append(metrics)

            # SHAP analysis (only for random split + best model to save time)
            if split_name == "random" and model_name == "XGBoost":
                logger.info("Computing SHAP feature importance...")
                shap_explainer = SHAPExplainer(
                    model,
                    X_train[:100],  # Background sample
                    feature_cols
                )
                shap_values = shap_explainer.compute_shap_values(X_test[:200])

                if shap_values is not None:
                    shap_explainer.plot_summary(
                        shap_values,
                        X_test[:200],
                        figures_dir / f"shap_summary_{model_name}.png"
                    )
                    shap_explainer.plot_bar(
                        shap_values,
                        figures_dir / f"shap_bar_{model_name}.png"
                    )

                    importance_df = shap_explainer.get_feature_importance_table(shap_values)
                    importance_df.to_csv(tables_dir / f"shap_importance_{model_name}.csv", index=False)
                    logger.info(f"\nTop 5 Most Important Features (SHAP):\n{importance_df.head()}")

    # Save results
    results_df = pd.DataFrame(all_results)
    results_csv = tables_dir / "benchmark_results.csv"
    results_df.to_csv(results_csv, index=False)
    logger.info(f"\n\nBenchmark results saved to {results_csv}")

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("  BENCHMARK SUMMARY")
    logger.info("="*80)
    
    summary = results_df.groupby(['split_strategy', 'model'])[['RMSE', 'MAE', 'R2', 'conformal_coverage']].mean()
    print(summary)

    # Generate comparison visualizations
    generate_comparison_plots(results_df, figures_dir)

    logger.info("\n" + "="*80)
    logger.info("  BENCHMARK PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("="*80)


def generate_comparison_plots(results_df: pd.DataFrame, output_dir: Path):
    """
    Generate model comparison visualizations.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Figure: RMSE comparison across splits and models
    fig, ax = plt.subplots(figsize=(12, 6))
    
    pivot_rmse = results_df.pivot(index='split_strategy', columns='model', values='RMSE')
    pivot_rmse.plot(kind='bar', ax=ax, width=0.8, edgecolor='white', linewidth=1.5)
    
    ax.set_title("Model Performance Comparison: RMSE Across Split Strategies", 
                 fontweight='bold', fontsize=14)
    ax.set_xlabel("Split Strategy", fontsize=12, fontweight='600')
    ax.set_ylabel("RMSE (% Degradation)", fontsize=12, fontweight='600')
    ax.legend(title="Model", title_fontsize=11, fontsize=10, frameon=True, fancybox=False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison_rmse.png", dpi=300, facecolor='white')
    plt.close()

    # Figure: R² comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    pivot_r2 = results_df.pivot(index='split_strategy', columns='model', values='R2')
    pivot_r2.plot(kind='bar', ax=ax, width=0.8, edgecolor='white', linewidth=1.5)
    
    ax.set_title("Model Performance Comparison: R² Across Split Strategies", 
                 fontweight='bold', fontsize=14)
    ax.set_xlabel("Split Strategy", fontsize=12, fontweight='600')
    ax.set_ylabel("R² Score", fontsize=12, fontweight='600')
    ax.legend(title="Model", title_fontsize=11, fontsize=10, frameon=True, fancybox=False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison_r2.png", dpi=300, facecolor='white')
    plt.close()

    # Figure: Conformal prediction coverage
    fig, ax = plt.subplots(figsize=(12, 6))
    
    pivot_coverage = results_df.pivot(index='split_strategy', columns='model', values='conformal_coverage')
    pivot_coverage.plot(kind='bar', ax=ax, width=0.8, edgecolor='white', linewidth=1.5)
    ax.axhline(y=0.90, color='red', linestyle='--', linewidth=2, label='Target 90% Coverage')
    
    ax.set_title("Conformal Prediction Coverage Across Splits", 
                 fontweight='bold', fontsize=14)
    ax.set_xlabel("Split Strategy", fontsize=12, fontweight='600')
    ax.set_ylabel("Empirical Coverage", fontsize=12, fontweight='600')
    ax.legend(title="Model", title_fontsize=11, fontsize=10, frameon=True, fancybox=False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "conformal_coverage.png", dpi=300, facecolor='white')
    plt.close()

    logger.info(f"Comparison plots saved to {output_dir}")


if __name__ == "__main__":
    main()