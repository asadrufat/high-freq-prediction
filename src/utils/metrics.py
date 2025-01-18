from typing import Dict, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

def calculate_metrics(y_true: np.ndarray, 
                     y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate common regression metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Dictionary containing various metrics
    """
    mse = np.mean(np.square(y_true - y_pred))
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'mse': mse
    }

def save_results(metrics: Dict[str, float],
                multipliers: np.ndarray,
                date: str,
                output_path: str) -> None:
    """
    Save model results and multipliers to CSV files.
    
    Args:
        metrics: Dictionary of metric values
        multipliers: Array of prediction multipliers
        date: Current date string
        output_path: Path to save results
    """
    # Save metrics
    metrics_df = pd.DataFrame({
        'prediction_date': [date],
        'rmse': [metrics['rmse']],
        'mae': [metrics['mae']],
        'mape': [metrics['mape']],
        'weight_regression': [metrics.get('weights', {}).get('regression', None)],
        'weight_median': [metrics.get('weights', {}).get('median', None)]
    })
    
    # Save multipliers
    multipliers_df = pd.DataFrame({
        'index': range(1, len(multipliers) + 1),
        'prediction_date': date,
        'multiplier': multipliers
    })
    
    metrics_df.to_csv(f"{output_path}/metrics_{date}.csv", index=False)
    multipliers_df.to_csv(f"{output_path}/multipliers_{date}.csv", index=False)

def evaluate_predictions(y_true: np.ndarray,
                       y_pred: np.ndarray,
                       timestamps: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Evaluate predictions across different time periods.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        timestamps: Array of corresponding timestamps
        
    Returns:
        Nested dictionary of metrics by period
    """
    # Convert timestamps to datetime if they're strings
    if isinstance(timestamps[0], str):
        timestamps = pd.to_datetime(timestamps)
    
    # Initialize results dictionary
    results = {
        'overall': calculate_metrics(y_true, y_pred),
        'by_hour': {},
        'by_day': {},
        'by_weekday': {}
    }
    
    # Calculate metrics by hour
    hours = pd.DatetimeIndex(timestamps).hour
    for hour in sorted(set(hours)):
        mask = hours == hour
        if mask.sum() > 0:
            results['by_hour'][hour] = calculate_metrics(
                y_true[mask], y_pred[mask]
            )
    
    # Calculate metrics by day
    days = pd.DatetimeIndex(timestamps).date
    for day in sorted(set(days)):
        mask = pd.DatetimeIndex(timestamps).date == day
        if mask.sum() > 0:
            results['by_day'][str(day)] = calculate_metrics(
                y_true[mask], y_pred[mask]
            )
    
    # Calculate metrics by weekday
    weekdays = pd.DatetimeIndex(timestamps).weekday
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                     'Friday', 'Saturday', 'Sunday']
    for weekday in range(7):
        mask = weekdays == weekday
        if mask.sum() > 0:
            results['by_weekday'][weekday_names[weekday]] = calculate_metrics(
                y_true[mask], y_pred[mask]
            )
    
    return results

def generate_prediction_report(evaluation_results: Dict,
                             output_path: str,
                             date: str) -> None:
    """
    Generate a comprehensive prediction report.
    
    Args:
        evaluation_results: Results from evaluate_predictions
        output_path: Path to save the report
        date: Current date string
    """
    report = []
    
    # Overall metrics
    report.append("# Prediction Evaluation Report")
    report.append(f"Generated on: {date}\n")
    
    report.append("## Overall Metrics")
    for metric, value in evaluation_results['overall'].items():
        report.append(f"- {metric.upper()}: {value:.4f}")
    
    # Metrics by period
    for period in ['by_hour', 'by_day', 'by_weekday']:
        report.append(f"\n## Metrics {period.replace('_', ' ').title()}")
        for key, metrics in evaluation_results[period].items():
            report.append(f"\n### {key}")
            for metric, value in metrics.items():
                report.append(f"- {metric.upper()}: {value:.4f}")
    
    # Save report
    with open(f"{output_path}/prediction_report_{date}.md", 'w') as f:
        f.write('\n'.join(report))

def calculate_business_impact_metrics(y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   cost_per_miss: float = 1.0) -> Dict[str, float]:
    """
    Calculate business-oriented impact metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        cost_per_miss: Cost factor for prediction errors
        
    Returns:
        Dictionary of business impact metrics
    """
    # Calculate over/under prediction
    over_prediction = np.maximum(y_pred - y_true, 0)
    under_prediction = np.maximum(y_true - y_pred, 0)
    
    # Calculate costs
    over_prediction_cost = np.sum(over_prediction) * cost_per_miss
    under_prediction_cost = np.sum(under_prediction) * cost_per_miss * 1.5  # Higher penalty for under-prediction
    
    return {
        'over_prediction_total': float(np.sum(over_prediction)),
        'under_prediction_total': float(np.sum(under_prediction)),
        'over_prediction_cost': float(over_prediction_cost),
        'under_prediction_cost': float(under_prediction_cost),
        'total_cost': float(over_prediction_cost + under_prediction_cost),
        'average_cost_per_period': float((over_prediction_cost + under_prediction_cost) / len(y_true))
    }