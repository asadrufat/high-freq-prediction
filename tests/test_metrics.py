from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.utils.metrics import (
    calculate_business_impact_metrics,
    calculate_metrics,
    evaluate_predictions,
)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)

    # Create sample timestamps
    base = datetime(2024, 1, 1)
    timestamps = [base + timedelta(hours=x) for x in range(100)]

    # Create sample true values and predictions
    y_true = np.random.normal(100, 10, 100)
    y_pred = y_true + np.random.normal(0, 5, 100)

    return y_true, y_pred, timestamps


def test_calculate_metrics(sample_data):
    """Test basic metric calculations."""
    y_true, y_pred, _ = sample_data

    metrics = calculate_metrics(y_true, y_pred)

    assert isinstance(metrics, dict)
    assert all(metric in metrics for metric in ["rmse", "mae", "mape", "mse"])
    assert all(isinstance(value, float) for value in metrics.values())
    assert all(value >= 0 for value in metrics.values())


def test_evaluate_predictions(sample_data):
    """Test prediction evaluation across different time periods."""
    y_true, y_pred, timestamps = sample_data

    results = evaluate_predictions(y_true, y_pred, timestamps)

    assert isinstance(results, dict)
    assert all(key in results for key in ["overall", "by_hour", "by_day", "by_weekday"])
    assert isinstance(results["overall"], dict)
    assert len(results["by_hour"]) > 0
    assert len(results["by_day"]) > 0
    assert len(results["by_weekday"]) == 7


def test_business_impact_metrics(sample_data):
    """Test business impact metric calculations."""
    y_true, y_pred, _ = sample_data

    metrics = calculate_business_impact_metrics(y_true, y_pred, cost_per_miss=1.0)

    assert isinstance(metrics, dict)
    assert all(
        metric in metrics
        for metric in [
            "over_prediction_total",
            "under_prediction_total",
            "over_prediction_cost",
            "under_prediction_cost",
            "total_cost",
            "average_cost_per_period",
        ]
    )
    assert all(isinstance(value, float) for value in metrics.values())
    assert all(value >= 0 for value in metrics.values())


def test_perfect_predictions():
    """Test metrics with perfect predictions."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])

    metrics = calculate_metrics(y_true, y_pred)

    assert metrics["rmse"] == 0
    assert metrics["mae"] == 0
    assert metrics["mse"] == 0
    assert metrics["mape"] == 0


def test_edge_cases():
    """Test edge cases and potential error conditions."""
    # Test with very small true values
    y_true = np.array([0.0001, 0.0002, 0.0003])
    y_pred = np.array([0.0002, 0.0003, 0.0004])

    metrics = calculate_metrics(y_true, y_pred)
    assert all(np.isfinite(value) for value in metrics.values())

    # Test with single value
    y_true = np.array([1])
    y_pred = np.array([1.1])

    metrics = calculate_metrics(y_true, y_pred)
    assert all(np.isfinite(value) for value in metrics.values())
