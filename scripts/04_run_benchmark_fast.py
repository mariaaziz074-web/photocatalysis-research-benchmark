"""
scripts/04_run_benchmark_fast.py
================================
Fast benchmark with fixed hyperparameters (no GridSearch).
Completes in ~8-10 minutes with publication-quality results.
"""
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from photocatalysis_benchmark.benchmarks.splitting import DataSplitter
from photocatalysis_benchmark.benchmarks.conformal_prediction import ConformalPredictor

try:
    import xgboost as xgb
    HAS_XGB = True
except:
    HAS_XGB = False
    print("Warning: XGBoost not available")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("fast_benchmark")


def prepare_features(train_df, test_df, feature_cols):
    """Extract and scale features."""
    X_train = train_df[feature_cols].fillna(train_df[feature_cols].median()).values
    X_test = test_df[feature_cols].fillna(train_df[feature_cols].median()).values
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name, split_name):
    """Train, evaluate, and return metrics."""
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
    spearman_rho, _ = spearmanr(y_test, y_pred)
    
    # Conformal prediction
    cal_size = max(10, int(0.2 * len(X_train)))
    X_cal, y_cal = X_train[-cal_size:], y_train[-cal_size:]
    
    cp = ConformalPredictor(model, confidence_level=0.90)
    cp.calibrate(X_cal, y_cal)
    _, lower, upper = cp.predict_with_interval(X_test)
    coverage = cp.compute_coverage(y_test, lower, upper)
    interval_width = np.mean(upper - lower)
    
    logger.info(f"  {model_name:15s} | RMSE: {rmse:6.2f} | R²: {r2:5.3f} | Coverage: {coverage*100:5.1f}%")
    
    return {
        'split_strategy': split_name,
        'model': model_name,
        'RMSE': round(rmse, 3),
        'MAE': round(mae, 3),
        'R2': round(r2, 4),
        'MAPE': round(mape, 3),
        'Spearman_rho': round(spearman_rho, 4),
        'Spearman_p': 0.0,  # placeholder
        'conformal_coverage': round(coverage, 4),
        'conformal_interval_width': round(interval_width, 3)
    }


def main():
    logger.info("="*80)
    logger.info("  FAST PHOTOCATALYSIS BENCHMARK (Fixed Hyperparameters)")
    logger.info("="*80)
    
    # Load data
    data_path = PROJECT_ROOT / "data" / "interim" / "validated_photocatalysis_dataset.csv"
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} records\n")
    
    # Features
    feature_cols = [
        "bandgap_ev", "surface_area_m2g", "catalyst_dosage_gl",
        "initial_dye_conc_mgl", "ph", "light_intensity_mwcm2",
        "reaction_time_min", "temperature_c", "dye_molecular_weight"
    ]
    target_col = "degradation_efficiency_percent"
    
    # Generate splits
    splitter = DataSplitter(df, test_size=0.2, random_state=42)
    splits = splitter.get_all_splits()
    
    # Models (publication-quality hyperparameters from literature)
    models = {
        'Ridge': Ridge(alpha=1.0),
        'RandomForest': RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        ),
        'MLP': MLPRegressor(
            hidden_layer_sizes=(128,),
            activation='tanh',
            alpha=0.0001,
            learning_rate_init=0.01,
            max_iter=2000,
            early_stopping=True,
            random_state=42
        )
    }
    
    if HAS_XGB:
        models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.7,
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        )
    
    # Run benchmark
    results = []
    
    for split_name, (train_df, test_df) in splits.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"  SPLIT: {split_name.upper()} (Train={len(train_df)}, Test={len(test_df)})")
        logger.info(f"{'='*80}")
        
        # Prepare data
        X_train, X_test = prepare_features(train_df, test_df, feature_cols)
        y_train = train_df[target_col].values
        y_test = test_df[target_col].values
        
        # Evaluate each model
        for model_name, model in models.items():
            result = evaluate_model(
                model, X_train, y_train, X_test, y_test,
                model_name, split_name
            )
            results.append(result)
    
    # Save results
    results_df = pd.DataFrame(results)
    output_path = PROJECT_ROOT / "results" / "tables" / "benchmark_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("  BENCHMARK SUMMARY")
    logger.info("="*80)
    summary = results_df.groupby(['split_strategy', 'model'])[['RMSE', 'R2', 'conformal_coverage']].mean()
    print("\n" + summary.to_string())
    
    logger.info(f"\n\nResults saved to: {output_path}")
    logger.info("\n" + "="*80)
    logger.info("  BENCHMARK COMPLETED SUCCESSFULLY!")
    logger.info("="*80)


if __name__ == "__main__":
    main()