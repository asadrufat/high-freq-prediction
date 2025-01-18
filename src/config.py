from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path

@dataclass
class ModelConfig:
    """Configuration for model parameters."""
    n_splits: int = 5
    lags: Tuple[int, ...] = (-1, -2, -3, -4, -5, -6, -7, -8, -9, -10,
                            -11, -12, -13, -14, -15, -16, -17, -18, -19,
                            -20, -21, -22, -23, -24, -48, -96)
    catboost_params: dict = {
        'iterations': 10000,
        'early_stopping_rounds': 1000,
        'verbose': 200,
        'random_state': 42,
        'eval_metric': 'MAPE'
    }
    validation_window: int = 11
    mcmc_samples: int = 12000
    mcmc_tune: int = 6000

@dataclass
class PathConfig:
    """Configuration for project paths."""
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    output_dir: Path = Path("output")
    raw_data_path: Path = data_dir / "raw"
    processed_data_path: Path = data_dir / "processed"
    model_artifacts_path: Path = models_dir / "artifacts"

@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    time_features: List[str] = (
        'hour', 'day_of_week', 'month', 'day_of_month',
        'day_of_year', 'week_of_month', 'is_business_hour',
        'is_lunch_time'
    )
    cyclical_features: List[str] = (
        'hour', 'day_of_week', 'month', 'week_of_month'
    )
    business_hours: Tuple[int, int] = (9, 20)
    lunch_hours: Tuple[int, int] = (13, 14)

class Config:
    """Main configuration class combining all configs."""
    def __init__(
        self,
        env: str = "development",
        model_config: ModelConfig = ModelConfig(),
        path_config: PathConfig = PathConfig(),
        feature_config: FeatureConfig = FeatureConfig()
    ):
        self.env = env
        self.model = model_config
        self.paths = path_config
        self.features = feature_config
        
    def setup_paths(self):
        """Create necessary directories if they don't exist."""
        for path in [
            self.paths.data_dir,
            self.paths.models_dir,
            self.paths.output_dir,
            self.paths.raw_data_path,
            self.paths.processed_data_path,
            self.paths.model_artifacts_path
        ]:
            path.mkdir(parents=True, exist_ok=True)

# Default configuration
config = Config()