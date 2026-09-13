"""
src/photocatalysis_benchmark/benchmarks/conformal_prediction.py
==============================================================
Conformal prediction for uncertainty quantification with prediction intervals.
"""

import logging
from typing import Tuple, Any
import numpy as np

logger = logging.getLogger("conformal_prediction")


class ConformalPredictor:
    """
    Implements split conformal prediction for regression.
    """

    def __init__(self, model: Any, confidence_level: float = 0.90):
        """
        Args:
            model: Trained sklearn-compatible model
            confidence_level: Target coverage probability (default 90%)
        """
        self.model = model
        self.confidence_level = confidence_level
        self.quantile = None

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray):
        """
        Calibrate conformal predictor on calibration set.
        """
        y_pred_cal = self.model.predict(X_cal)
        residuals = np.abs(y_cal - y_pred_cal)

        # Compute quantile for desired coverage
        alpha = 1 - self.confidence_level
        n = len(residuals)
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        self.quantile = np.quantile(residuals, q_level)

        logger.info(f"Conformal quantile at {self.confidence_level*100}% coverage: {self.quantile:.3f}")

    def predict_with_interval(
        self, 
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions with prediction intervals.

        Returns:
            y_pred: Point predictions
            lower_bound: Lower confidence bound
            upper_bound: Upper confidence bound
        """
        if self.quantile is None:
            raise ValueError("Must calibrate predictor before making predictions")

        y_pred = self.model.predict(X_test)
        lower_bound = y_pred - self.quantile
        upper_bound = y_pred + self.quantile

        return y_pred, lower_bound, upper_bound

    def compute_coverage(
        self, 
        y_true: np.ndarray, 
        lower_bound: np.ndarray, 
        upper_bound: np.ndarray
    ) -> float:
        """
        Compute empirical coverage rate.
        """
        coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
        logger.info(f"Empirical coverage: {coverage*100:.2f}%")
        return coverage