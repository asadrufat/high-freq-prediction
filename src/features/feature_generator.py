from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

from ..config import Config, FeatureConfig


class FeatureGenerator:
    """Generates features for the call volume prediction model."""

    def __init__(self, config: Config):
        self.config = config
        self.feature_config = config.features

    def _sin_transformer(self, period: int) -> FunctionTransformer:
        """Create sinusoidal transformation of time features."""
        return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

    def _cos_transformer(self, period: int) -> FunctionTransformer:
        """Create cosinusoidal transformation of time features."""
        return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

    def _create_cyclical_features(self, df: pd.DataFrame, feature: str) -> pd.DataFrame:
        """Create cyclical features from time components."""
        df = df.copy()

        # Define periods for different features
        periods = {"hour": 24, "day_of_week": 7, "month": 12, "week_of_month": 5}

        if feature in periods:
            period = periods[feature]
            df[f"{feature}_sin"] = self._sin_transformer(period).fit_transform(
                df[feature]
            )
            df[f"{feature}_cos"] = self._cos_transformer(period).fit_transform(
                df[feature]
            )

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic time-based features."""
        df = df.copy()

        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek + 1
        df["month"] = df["timestamp"].dt.month
        df["day_of_month"] = df["timestamp"].dt.day
        df["day_of_year"] = df["timestamp"].dt.dayofyear
        df["week_of_month"] = df["timestamp"].dt.day.apply(lambda x: (x - 1) // 7 + 1)

        return df

    def _add_business_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add business-related features."""
        df = df.copy()

        # Business hours
        df["is_business_hour"] = (
            (df["day_of_week"] <= 5)
            & (df["hour"] >= self.feature_config.business_hours[0])
            & (df["hour"] <= self.feature_config.business_hours[1])
        ).astype(int)

        # Lunch time
        df["is_lunch_time"] = (
            (df["day_of_week"] <= 5)
            & (df["hour"] >= self.feature_config.lunch_hours[0])
            & (df["hour"] <= self.feature_config.lunch_hours[1])
        ).astype(int)

        return df

    def _add_special_day_features(
        self, df: pd.DataFrame, calendar_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Add features for holidays and special days."""
        df = df.copy()

        # Merge calendar data
        df["date"] = df["timestamp"].dt.date
        df = pd.merge(df, calendar_data, on="date", how="left")

        # Fill NaN values for holiday columns
        holiday_columns = ["is_holiday", "is_weekend", "is_special_day"]
        for col in holiday_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        return df

    def _add_lagged_features(self, df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
        """Add lagged features for the target variable."""
        df = df.copy()

        for lag in lags:
            df[f"call_volume_lag_{abs(lag)}"] = df["call_volume"].shift(lag)

        return df

    def generate_features(
        self,
        df: pd.DataFrame,
        calendar_data: pd.DataFrame = None,
        add_lags: bool = True,
    ) -> pd.DataFrame:
        """
        Generate all features for the model.

        Args:
            df: Input DataFrame
            calendar_data: Optional calendar data with holiday information
            add_lags: Whether to add lagged features

        Returns:
            DataFrame with all generated features
        """
        df = df.copy()

        # Add time-based features
        df = self._add_time_features(df)

        # Add cyclical features
        for feature in self.feature_config.cyclical_features:
            df = self._create_cyclical_features(df, feature)

        # Add business features
        df = self._add_business_features(df)

        # Add special day features if calendar data is provided
        if calendar_data is not None:
            df = self._add_special_day_features(df, calendar_data)

        # Add lagged features if requested
        if add_lags:
            df = self._add_lagged_features(df, list(self.config.model.lags))

        return df

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        features = self.feature_config.time_features + [
            f"{feat}_{transform}"
            for feat in self.feature_config.cyclical_features
            for transform in ["sin", "cos"]
        ]

        return features
