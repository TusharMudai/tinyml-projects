import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Set

class BatteryFeatureMapper:
    """
    Maps various column names from different battery datasets to standardized feature names.
    """
    
    def __init__(self):
        self.feature_patterns = {
            'capacity': [
                r'capacity', r'cap', r'q', r'charge_capacity', r'discharge_capacity',
                r'nominal_capacity', r'ah', r'ampere_hour', r'mah', r'milliampere'
            ],
            'voltage': [
                r'voltage', r'volt', r'v', r'batt_voltage', r'cell_voltage',
                r'terminal_voltage', r'v_batt', r'volts', r'voltage_v'
            ],
            'current': [
                r'current', r'curr', r'i', r'batt_current', r'cell_current',
                r'discharge_current', r'charge_current', r'amps', r'amperes',
                r'current_a', r'ampere'
            ],
            'temperature': [
                r'temperature', r'temp', r'tmp', r'batt_temp', r'cell_temp',
                r't_batt', r'thermal', r'deg', r'celsius', r'fahrenheit',
                r'temp_c', r'temp_f'
            ],
            'discharge_time': [
                r'discharge_time', r'disch_time', r't_discharge', r't_disch',
                r'discharge_duration', r'duration_discharge', r'disc_time',
                r'time_discharge'
            ],
            'charge_time': [
                r'charge_time', r'chg_time', r't_charge', r't_chg',
                r'charge_duration', r'duration_charge', r'ch_time',
                r'time_charge'
            ]
        }
        
        self.standard_features = list(self.feature_patterns.keys())
    
    def find_matching_columns(self, df_columns: List[str]) -> Dict[str, str]:
        """
        Find columns that match our feature patterns.
        
        Args:
            df_columns: List of column names from the dataset
            
        Returns:
            Dictionary mapping standard feature names to actual column names
        """
        matches = {}
        used_columns = set()
        
        for feature, patterns in self.feature_patterns.items():
            for pattern in patterns:
                # Case-insensitive search
                regex = re.compile(pattern, re.IGNORECASE)
                for col in df_columns:
                    if (regex.search(col) and 
                        col not in used_columns and 
                        feature not in matches):
                        matches[feature] = col
                        used_columns.add(col)
                        break
                if feature in matches:
                    break
        
        return matches
    
    def extract_features(self, df: pd.DataFrame, mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Extract and standardize features from the dataframe.
        
        Args:
            df: Input dataframe
            mapping: Optional manual mapping of standard features to column names
            
        Returns:
            Dataframe with standardized feature names
        """
        if mapping is None:
            mapping = self.find_matching_columns(df.columns.tolist())
        
        result_df = pd.DataFrame()
        
        for std_feature, actual_col in mapping.items():
            if actual_col in df.columns:
                result_df[std_feature] = df[actual_col]
            else:
                print(f"Warning: Column '{actual_col}' not found in dataframe")
        
        # Report missing features
        missing_features = set(self.standard_features) - set(mapping.keys())
        if missing_features:
            print(f"Missing features: {missing_features}")
        
        return result_df
    
    def get_available_features(self, df_columns: List[str]) -> Set[str]:
        """
        Get set of available features in the dataset.
        """
        mapping = self.find_matching_columns(df_columns)
        return set(mapping.keys())
    
    def add_custom_pattern(self, feature_name: str, patterns: List[str]):
        """
        Add custom pattern for a new feature.
        
        Args:
            feature_name: Name of the feature
            patterns: List of regex patterns to match
        """
        self.feature_patterns[feature_name] = patterns
        self.standard_features = list(self.feature_patterns.keys())
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """
        Handle missing values in the extracted features.
        
        Args:
            df: Dataframe with standardized features
            strategy: Strategy for handling missing values ('mean', 'median', 'drop', 'forward_fill')
            
        Returns:
            Dataframe with handled missing values
        """
        df_clean = df.copy()
        
        if strategy == 'drop':
            df_clean = df_clean.dropna()
        elif strategy == 'mean':
            df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))
        elif strategy == 'median':
            df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
        elif strategy == 'forward_fill':
            df_clean = df_clean.fillna(method='ffill')
        elif strategy == 'backward_fill':
            df_clean = df_clean.fillna(method='bfill')
        else:
            print(f"Warning: Unknown strategy '{strategy}'. Using 'mean' instead.")
            df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))
        
        print(f"Missing values handled using '{strategy}' strategy")
        print(f"Original shape: {df.shape}, After handling: {df_clean.shape}")
        print(f"Missing values before: {df.isnull().sum().sum()}, after: {df_clean.isnull().sum().sum()}")
        
        return df_clean