# Call Volume Predictor

A machine learning system for predicting call center volume using time series analysis and Bayesian optimization. The system combines CatBoost-based prediction with MCMC optimization to provide accurate call volume forecasts.

## Features

- Time series-based call volume prediction using CatBoost
- Advanced feature engineering for datetime data
- Bayesian optimization using MCMC
- Robust evaluation metrics and model diagnostics
- Support for holiday and special event handling
- Business hour and seasonality awareness

## Project Structure

high-freq-prediction/
├── src/                    # Source code
│   ├── data/              # Data loading and preprocessing
│   ├── features/          # Feature engineering
│   ├── models/            # ML models
│   └── utils/             # Utility functions
├── tests/                 # Unit tests
├── notebooks/            # Jupyter notebooks
└── configs/              # Configuration files

## Installation