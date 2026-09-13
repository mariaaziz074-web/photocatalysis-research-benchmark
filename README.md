# Photocatalysis Research Benchmark

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Publication-quality photocatalysis dataset and ML benchmark for dye degradation studies.**

## 🎯 Objective

Create a curated, provenance-tracked dataset of photocatalytic dye degradation experiments with rigorous ML benchmarking demonstrating proper evaluation methodology.

**Target Publication:** Scientific Data (Nature Portfolio)

## 📊 Dataset Overview

- **Size:** 500+ validated data points
- **Sources:** 50+ peer-reviewed papers (2015-2024)
- **Coverage:** TiO₂, ZnO, g-C₃N₄ photocatalysts
- **Pollutants:** Methylene Blue, Rhodamine B, Methyl Orange, and others
- **Provenance:** 100% traceable to source DOI/table/page

## 🔬 Features

### Data Quality
✅ Chemical validation (formulas, structures)  
✅ Unit normalization (concentration, time, energy)  
✅ Outlier detection  
✅ Duplicate detection  
✅ Quality tier assignment  

### ML Benchmark
✅ 5 evaluation strategies (random, catalyst-holdout, dye-holdout, paper-holdout, time-based)  
✅ 4 model types (Ridge, Random Forest, XGBoost, Neural Network)  
✅ Uncertainty quantification  
✅ Feature importance analysis (SHAP)  

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/mariaaziz074-web/photocatalysis-research-benchmark.git
cd photocatalysis-research-benchmark

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install package
pip install -e .