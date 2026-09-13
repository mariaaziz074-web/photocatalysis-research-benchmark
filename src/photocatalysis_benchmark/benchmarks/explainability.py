"""
src/photocatalysis_benchmark/benchmarks/explainability.py
========================================================
SHAP-based feature importance and model explainability.
"""

import logging
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

logger = logging.getLogger("explainability")


class SHAPExplainer:
    """
    SHAP-based model interpretation.
    """

    def __init__(self, model: Any, X_background: np.ndarray, feature_names: list):
        """
        Args:
            model: Trained model
            X_background: Background dataset for SHAP (typically training set sample)
            feature_names: List of feature names
        """
        if not HAS_SHAP:
            logger.warning("SHAP not installed. Install with: pip install shap")
            self.explainer = None
            return

        self.model = model
        self.feature_names = feature_names

        # Use appropriate explainer based on model type
        model_type = type(model).__name__

        if 'RandomForest' in model_type or 'XGB' in model_type:
            # Tree-based explainer (faster)
            self.explainer = shap.TreeExplainer(model)
        else:
            # Kernel explainer (model-agnostic, slower)
            # Sample background to speed up
            if len(X_background) > 100:
                background_sample = shap.sample(X_background, 100)
            else:
                background_sample = X_background
            self.explainer = shap.KernelExplainer(model.predict, background_sample)

    def compute_shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for dataset.
        """
        if self.explainer is None:
            logger.warning("SHAP explainer not available")
            return None

        logger.info(f"Computing SHAP values for {len(X)} samples...")
        shap_values = self.explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        return shap_values

    def plot_summary(
        self, 
        shap_values: np.ndarray, 
        X: np.ndarray, 
        output_path: Path,
        max_display: int = 15
    ):
        """
        Generate SHAP summary plot (beeswarm).
        """
        if shap_values is None:
            return

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=self.feature_names,
            show=False,
            max_display=max_display
        )
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, facecolor='white', bbox_inches='tight')
        plt.close()
        logger.info(f"SHAP summary plot saved to {output_path}")

    def plot_bar(
        self, 
        shap_values: np.ndarray, 
        output_path: Path,
        max_display: int = 15
    ):
        """
        Generate SHAP bar plot (mean absolute SHAP values).
        """
        if shap_values is None:
            return

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            feature_names=self.feature_names,
            plot_type="bar",
            show=False,
            max_display=max_display
        )
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, facecolor='white', bbox_inches='tight')
        plt.close()
        logger.info(f"SHAP bar plot saved to {output_path}")

    def get_feature_importance_table(self, shap_values: np.ndarray) -> pd.DataFrame:
        """
        Extract feature importance as DataFrame.
        """
        if shap_values is None:
            return pd.DataFrame()

        importance = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Mean_Abs_SHAP': importance
        }).sort_values('Mean_Abs_SHAP', ascending=False)

        return importance_df