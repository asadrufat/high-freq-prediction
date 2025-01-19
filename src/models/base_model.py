import json
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Abstract base class for all prediction models."""

    def __init__(self, config: Any):
        self.config = config
        self.model = None
        self.feature_names = None

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, float]:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training target values

        Returns:
            Dictionary of training metrics
        """
        pass

    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions.

        Args:
            X_test: Test features

        Returns:
            Array of predictions
        """
        pass

    @abstractmethod
    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            y_true: True target values
            y_pred: Predicted values

        Returns:
            Dictionary of evaluation metrics
        """
        pass

    def save(self, path: Path) -> None:
        """
        Save model and metadata.

        Args:
            path: Path to save model
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model
        with open(path / "model.pkl", "wb") as f:
            pickle.dump(self.model, f)

        # Save metadata
        metadata = {"feature_names": self.feature_names, "config": self.config.__dict__}

        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

    def load(self, path: Path) -> None:
        """
        Load model and metadata.

        Args:
            path: Path to load model from
        """
        path = Path(path)

        # Load model
        with open(path / "model.pkl", "rb") as f:
            self.model = pickle.load(f)

        # Load metadata
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)
            self.feature_names = metadata["feature_names"]
