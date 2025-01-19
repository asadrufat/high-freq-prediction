# High Frequency Prediction Model

A machine learning system for predicting call center volume using time series analysis and Bayesian optimization. The system combines CatBoost-based prediction with MCMC optimization to provide accurate call volume forecasts.

## Features

- Time series-based call volume prediction using CatBoost
- Advanced feature engineering for datetime data
- Bayesian optimization using MCMC
- Robust evaluation metrics and model diagnostics
- Support for holiday and special event handling
- Business hour and seasonality awareness

## Project Structure

```
high-freq-prediction/
├── src/                  # Source code
│   ├── data/               # Data loading and preprocessing
│   ├── features/           # Feature engineering
│   ├── models/             # ML models
│   └── utils/              # Utility functions
├── tests/                # Unit tests
├── notebooks/            # Jupyter notebooks
└── configs/              # Configuration files
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/high-freq-prediction.git
cd high-freq-prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.config import Config
from src.data import DataLoader
from src.features import FeatureGenerator
from src.models import TimeSeriesCallPredictor

# Initialize configuration
config = Config()

# Load and prepare data
loader = DataLoader(config)
train_data, test_data = loader.prepare_data_for_training()

# Generate features
feature_gen = FeatureGenerator(config)
train_features = feature_gen.generate_features(train_data)
test_features = feature_gen.generate_features(test_data)

# Train model
model = TimeSeriesCallPredictor(config)
metrics = model.train(train_features, train_data['call_volume'])

# Make predictions
predictions = model.predict(test_features)
```

## Model Details

### Architecture

The system uses a two-stage approach for prediction:

1. **Base Prediction**:
   - CatBoost model with time series features
   - Handles high-frequency temporal patterns
   - Incorporates business-specific features

2. **MCMC Optimization**:
   - Bayesian refinement of predictions
   - Accounts for uncertainty
   - Adapts to changing patterns

### Key Components

- **Feature Engineering**
  - Time-based features (hour, day, month)
  - Cyclical transformations
  - Business period indicators
  - Holiday and special event handling

- **Model Training**
  - Time series cross-validation
  - Ensemble predictions
  - Automated hyperparameter tuning

## Configuration

Configuration is managed through `src/config.py`:

```python
@dataclass
class ModelConfig:
    n_splits: int = 5
    validation_window: int = 11
    mcmc_samples: int = 12000
    mcmc_tune: int = 6000
```

Key configurations include:
- Model hyperparameters
- Feature settings
- MCMC parameters
- Path configurations

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Code Style

The project follows PEP 8 guidelines. To check:

```bash
flake8 src tests
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
