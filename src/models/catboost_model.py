from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from darts import TimeSeries
from darts.models import RegressionModel
from sklearn.model_selection import TimeSeriesSplit
from .base_model import BaseModel
from ..config import Config

class TimeSeriesCallPredictor(BaseModel):
    """CatBoost-based time series prediction model for call volume."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.models = []  # List to store models for each fold
        self.tss = TimeSeriesSplit(n_splits=config.model.n_splits)
        
    def _prepare_time_series(
        self,
        X: pd.DataFrame,
        y: pd.Series = None
    ) -> Tuple[TimeSeries, TimeSeries]:
        """Convert pandas objects to Darts TimeSeries."""
        feature_cov = TimeSeries.from_series(X)
        if y is not None:
            y_ts = TimeSeries.from_series(y)
            return feature_cov, y_ts
        return feature_cov, None
        
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Dict[str, float]:
        """
        Train multiple CatBoost models using time series cross-validation.
        
        Args:
            X_train: Training features
            y_train: Training target values
            
        Returns:
            Dictionary of training metrics
        """
        self.feature_names = X_train.columns.tolist()
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(self.tss.split(X_train)):
            # Prepare fold data
            X_fold = X_train.iloc[train_idx]
            y_fold = y_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx]
            y_val = y_train.iloc[val_idx]
            
            # Convert to TimeSeries
            feature_cov_train, y_train_ts = self._prepare_time_series(X_fold, y_fold)
            feature_cov_val, _ = self._prepare_time_series(X_val)
            
            # Initialize and train model
            model = RegressionModel(
                lags=self.config.model.lags,
                lags_future_covariates=[0],
                model=CatBoostRegressor(**self.config.model.catboost_params)
            )
            
            model.fit(y_train_ts, future_covariates=feature_cov_train)
            self.models.append(model)
            
            # Evaluate fold
            preds = model.predict(
                n=len(val_idx),
                series=y_train_ts,
                future_covariates=feature_cov_val
            )
            score = np.mean(np.abs((y_val - preds.values()) / y_val)) * 100
            scores.append(score)
            
            print(f"Fold {fold + 1} MAPE: {score:.2f}%")
        
        return {
            'mape_scores': scores,
            'mean_mape': np.mean(scores),
            'std_mape': np.std(scores)
        }

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Generate ensemble predictions from all trained models.
        
        Args:
            X_test: Test features
            
        Returns:
            Array of predictions
        """
        if not self.models:
            raise ValueError("Model has not been trained yet")
            
        feature_cov_test, _ = self._prepare_time_series(X_test)
        predictions = []
        
        for model in self.models:
            pred = model.predict(
                n=len(X_test),
                future_covariates=feature_cov_test
            )
            predictions.append(pred.values())
            
        return np.mean(predictions, axis=0)

    def evaluate(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate various model performance metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Dictionary of evaluation metrics
        """
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        return {
            'mape': mape,
            'mae': mae,
            'rmse': rmse
        }
        
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the models.
        
        Returns:
            DataFrame with feature importance scores
        """
        if not self.models or not self.feature_names:
            raise ValueError("Model has not been trained yet")
        
        # Get feature importance from each model and average them
        importance_scores = []
        for model in self.models:
            # Access the underlying CatBoost model
            catboost_model = model.model.regressor_
            importance_scores.append(catboost_model.get_feature_importance())
            
        avg_importance = np.mean(importance_scores, axis=0)
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': avg_importance
        }).sort_values('importance', ascending=False)