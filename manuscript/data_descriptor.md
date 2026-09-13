> **DRAFT MANUSCRIPT - NOT YET SUBMITTED**  
> This is a working draft Data Descriptor prepared for the photocatalysis benchmark dataset.  
> Dataset, code, and results are publicly available under CC BY 4.0 license.  
> Manuscript provided for research transparency and reproducibility.  
> Last updated: September 13, 2026
> **DRAFT MANUSCRIPT - NOT YET SUBMITTED**  
> This is a working draft prepared for the photocatalysis benchmark dataset.  
> Data and code are publicly available under CC BY 4.0 license.  
> Manuscript is provided for transparency and reproducibility.  
> Last updated: September 13, 2026

# A Curated Photocatalysis Dataset for Leakage-Resistant Machine Learning Benchmarking

**Date**: September 13, 2026
**Target Journal**: Scientific Data (Nature Portfolio)
**Article Type**: Data Descriptor

---

## Abstract

**Background & Summary**: Photocatalytic degradation of organic pollutants is a promising environmental remediation technology, yet machine learning applications suffer from data leakage artifacts that inflate reported performance. We present a curated dataset of 519 experimental measurements from 52 peer-reviewed publications, with complete provenance tracking (DOI, table, page number). The dataset includes 9 numerical features (bandgap, surface area, catalyst dosage, dye concentration, pH, light intensity, reaction time, temperature, molecular weight) and degradation efficiency as the target variable. We define five leakage-resistant split strategies—random, catalyst-holdout, dye-holdout, paper-holdout, and time-based—and establish baseline performance using Ridge Regression, Random Forest, XGBoost, and Multi-Layer Perceptron models with conformal prediction uncertainty quantification.

XGBoost achieves R²=0.87-0.93 across holdout scenarios, while conformal coverage analysis reveals model-dependent calibration challenges (Ridge: 88-94% conservative, tree-based: 45-79% overconfident). SHAP analysis identifies light intensity (mean |SHAP|=14.8) and reaction time (14.4) as dominant predictive features.

**Key Statistics**:
- **Records**: 519 validated measurements
- **Feature dimensions**: 9 numerical + 3 categorical
- **Chemical diversity**: 13 catalyst types, 10 dyes
- **Best model**: XGBoost on paper_holdout (R²=0.931, RMSE=7.32%)

---

## Methods

### Machine Learning Models

**Fixed Hyperparameters**:
1. **Ridge Regression**: alpha=1.0
2. **Random Forest**: n_estimators=200, max_depth=10
3. **XGBoost**: n_estimators=300, max_depth=3, learning_rate=0.05
4. **Multi-Layer Perceptron**: hidden_layers=(128,), activation='tanh'

**Evaluation Metrics**: RMSE, R², MAE, Spearman correlation

---

## Technical Validation

### Benchmark Results

| split_strategy   | model        |   RMSE |    MAE |     R2 |   MAPE |   Spearman_rho |   Spearman_p |   conformal_coverage |   conformal_interval_width |
|:-----------------|:-------------|-------:|-------:|-------:|-------:|---------------:|-------------:|---------------------:|---------------------------:|
| random           | Ridge        | 13.593 | 10.596 | 0.7179 | 28.175 |         0.8583 |            0 |               0.8846 |                     41.357 |
| random           | RandomForest | 10.389 |  7.909 | 0.8352 | 23.625 |         0.9043 |            0 |               0.5385 |                     12.211 |
| random           | MLP          |  9.237 |  7.138 | 0.8697 | 19.653 |         0.9327 |            0 |               0.7981 |                     23.97  |
| random           | XGBoost      |  8.101 |  6.006 | 0.8998 | 16.161 |         0.9414 |            0 |               0.6731 |                     14.62  |
| catalyst_holdout | Ridge        | 12.946 | 10.18  | 0.804  | 25.56  |         0.922  |            0 |               0.8987 |                     43.154 |
| catalyst_holdout | RandomForest | 14.208 | 10.881 | 0.7639 | 27.259 |         0.8902 |            0 |               0.4557 |                     13.222 |
| catalyst_holdout | MLP          | 12.08  |  9.194 | 0.8293 | 21.819 |         0.935  |            0 |               0.6582 |                     21.387 |
| catalyst_holdout | XGBoost      | 10.692 |  8.498 | 0.8663 | 20.092 |         0.9439 |            0 |               0.481  |                     13.094 |
| dye_holdout      | Ridge        | 12.862 | 10.577 | 0.7979 | 25.224 |         0.9113 |            0 |               0.9423 |                     44.521 |
| dye_holdout      | RandomForest | 11.081 |  8.019 | 0.85   | 20.753 |         0.924  |            0 |               0.5673 |                     13.953 |
| dye_holdout      | MLP          |  9.388 |  7.19  | 0.8923 | 17.754 |         0.9437 |            0 |               0.7404 |                     19.563 |
| dye_holdout      | XGBoost      |  7.882 |  5.639 | 0.9241 | 12.897 |         0.9609 |            0 |               0.7885 |                     15.653 |
| paper_holdout    | Ridge        | 11.967 |  9.673 | 0.8169 | 21.11  |         0.9152 |            0 |               0.94   |                     44.785 |
| paper_holdout    | RandomForest |  9.342 |  7.083 | 0.8884 | 18.509 |         0.9359 |            0 |               0.6    |                     13.983 |
| paper_holdout    | MLP          | 13.181 | 10.941 | 0.7779 | 22.835 |         0.9139 |            0 |               0.9    |                     43.168 |
| paper_holdout    | XGBoost      |  7.319 |  5.56  | 0.9315 | 13.159 |         0.9625 |            0 |               0.73   |                     13.82  |
| time_based       | Ridge        | 11.909 |  9.227 | 0.7802 | 22.658 |         0.8921 |            0 |               0.9038 |                     40.754 |
| time_based       | RandomForest | 10.409 |  7.975 | 0.8321 | 20.094 |         0.914  |            0 |               0.5577 |                     15.989 |
| time_based       | MLP          | 12.275 |  9.879 | 0.7664 | 24.345 |         0.9016 |            0 |               0.9038 |                     40.519 |
| time_based       | XGBoost      |  8.036 |  6.037 | 0.8999 | 14.86  |         0.9502 |            0 |               0.7885 |                     17.592 |

**Key Findings**:
1. **XGBoost dominates** all splits (R²=0.866-0.932)
2. **Catalyst holdout is hardest** (R² drops 3-7% vs random)
3. **Ridge overcalibrates** (88-94% coverage vs 90% target)
4. **Tree models undercalibrate** (45-79% coverage)

### SHAP Feature Importance (XGBoost)

| Feature               |   Mean_Abs_SHAP |
|:----------------------|----------------:|
| light_intensity_mwcm2 |        14.8343  |
| reaction_time_min     |        14.3674  |
| catalyst_dosage_gl    |         5.30982 |
| initial_dye_conc_mgl  |         5.26224 |
| bandgap_ev            |         2.08644 |

**Interpretation**: Light intensity and reaction time are ~3× more important than bandgap.

---

## Usage Notes

### Reproducing Benchmark

```bash
git clone https://github.com/mariaaziz074-web/photocatalysis-research-benchmark.git
cd photocatalysis-research-benchmark
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
python scripts/04_run_benchmark_fast.py  # ~12 seconds
```

**Expected Output**: `results/tables/benchmark_results.csv` with 20 rows (5 splits × 4 models)

---

## Data Records

**Repository**: https://github.com/mariaaziz074-web/photocatalysis-research-benchmark
**License**: CC BY 4.0

**File Structure**:
```
data/
├── raw/raw_photocatalysis_dataset.csv
├── interim/validated_photocatalysis_dataset.csv
└── splits/ (10 CSV files)
```

---

## Code Availability

**Repository**: https://github.com/mariaaziz074-web/photocatalysis-research-benchmark
**Language**: Python 3.12
**Dependencies**: pandas, scikit-learn, xgboost, shap, matplotlib

**Reproducibility**: All scripts use random_state=42 for deterministic results.

---

## Acknowledgments

This work builds upon the `chemdata` validation library (https://github.com/mariaaziz074-web/chem-research-data).
