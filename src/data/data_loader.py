from typing import Tuple, Optional
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from ..config import Config

class DataLoader:
    """
    Handles loading and basic preprocessing of call center data.
    
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.config.setup_paths()

    def load_call_data(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load main call volume dataset.
        
        Args:
            file_path: Optional specific path to data file
            
        Returns:
            DataFrame with call volume data
        """
        if file_path is None:
            file_path = self.config.paths.raw_data_path / "call_volume.csv"
            
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp').reset_index(drop=True)

    def load_calendar_data(self) -> pd.DataFrame:
        """
        Load calendar-related data (holidays, special dates).
        
        Returns:
            DataFrame with calendar data
        """
        file_path = self.config.paths.raw_data_path / "calendar_data.csv"
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        return df

    def split_train_test(
        self,
        df: pd.DataFrame,
        cutoff_date: Optional[str] = None,
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets.
        
        Args:
            df: Input DataFrame
            cutoff_date: Optional specific cutoff date for splitting
            test_size: Fraction of data to use for testing if no cutoff_date
            
        Returns:
            Tuple of (train_df, test_df)
        """
        if cutoff_date:
            mask = df['timestamp'] <= cutoff_date
        else:
            # Use the last test_size portion of the data
            cutoff_idx = int(len(df) * (1 - test_size))
            mask = df.index <= cutoff_idx
            
        return df[mask], df[~mask]

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform basic preprocessing on the data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        df = df.copy()
        
        # Handle missing values
        df['call_volume'] = df['call_volume'].fillna(0)
        
        # Remove extreme outliers (e.g., beyond 3 std from mean)
        mean = df['call_volume'].mean()
        std = df['call_volume'].std()
        df.loc[df['call_volume'] > mean + 3*std, 'call_volume'] = mean + 3*std
        
        return df

    def prepare_data_for_training(
        self,
        cutoff_date: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load, preprocess, and split data for training.
        
        Args:
            cutoff_date: Optional specific cutoff date for splitting
            
        Returns:
            Tuple of (train_df, test_df)
        """
        # Load data
        df = self.load_call_data()
        
        # Preprocess
        df = self.preprocess_data(df)
        
        # Split into train/test
        train_df, test_df = self.split_train_test(df, cutoff_date)
        
        return train_df, test_df