"""
src/photocatalysis_benchmark/benchmarks/models.py
================================================
Four baseline ML models with hyperparameter optimization.
"""

import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

logger = logging.getLogger("models")


class PhotocatalysisModelBenchmark:
    """
    ML model training and evaluation framework.
    """

    def __init__(self, feature_cols: list, target_col: str = "degradation_efficiency_percent"):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.scaler = StandardScaler()
        self.models = {}

    def prepare_data(
        self, 
        train_df: pd.DataFrame, 
        test_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract features, handle categorical encoding, and scale.
        """
        # Select only numerical features for now (categorical encoding will be added if needed)
        numerical_features = [
            col for col in self.feature_cols 
            if col in train_df.columns and pd.api.types.is_numeric_dtype(train_df[col])
        ]

        X_train = train_df[numerical_features].fillna(train_df[numerical_features].median())
        X_test = test_df[numerical_features].fillna(train_df[numerical_features].median())
        
        y_train = train_df[self.target_col].values
        y_test = test_df[self.target_col].values

        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        logger.info(f"Prepared data: X_train={X_train_scaled.shape}, X_test={X_test_scaled.shape}")
        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_ridge(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        cv: int = 5
    ) -> Ridge:
        """
        Ridge Regression with hyperparameter tuning.
        """
        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        }

        grid_search = GridSearchCV(
            Ridge(),
            param_grid,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        logger.info(f"Ridge best params: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def train_random_forest(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        cv: int = 3
    ) -> RandomForestRegressor:
        """
        Random Forest with hyperparameter tuning.
        """
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=-1),
            param_grid,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)

        logger.info(f"Random Forest best params: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def train_xgboost(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        cv: int = 3
    ) -> Any:
        """
        XGBoost with hyperparameter tuning.
        """
        if not HAS_XGBOOST:
            logger.warning("XGBoost not installed. Skipping.")
            return None

        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
        }

        grid_search = GridSearchCV(
            xgb.XGBRegressor(random_state=42, n_jobs=-1, tree_method='hist'),
            param_grid,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)

        logger.info(f"XGBoost best params: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def train_mlp(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        cv: int = 3
    ) -> MLPRegressor:
        """
        Multi-Layer Perceptron (Neural Network) with hyperparameter tuning.
        """
        param_grid = {
            'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64), (128, 64, 32)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01]
        }

        grid_search = GridSearchCV(
            MLPRegressor(random_state=42, max_iter=1000, early_stopping=True),
            param_grid,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)

        logger.info(f"MLP best params: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def evaluate_model(
        self, 
        model: Any, 
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.
        """
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
        spearman_rho, spearman_p = spearmanr(y_test, y_pred)

        return {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'Spearman_rho': spearman_rho,
            'Spearman_p': spearman_p
        }

    def train_all_models(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray
    ) -> Dict[str, Any]:
        """
        Train all 4 baseline models.
        """
        models = {}

        logger.info("Training Ridge Regression...")
        models['Ridge'] = self.train_ridge(X_train, y_train)

        logger.info("Training Random Forest...")
        models['RandomForest'] = self.train_random_forest(X_train, y_train)

        if HAS_XGBOOST:
            logger.info("Training XGBoost...")
            models['XGBoost'] = self.train_xgboost(X_train, y_train)

        logger.info("Training MLP Neural Network...")
        models['MLP'] = self.train_mlp(X_train, y_train)

        self.models = models
        return models