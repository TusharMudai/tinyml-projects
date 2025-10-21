import pandas as pd
import os
import re
from typing import List, Dict, Any
from battery_feature_mapper import BatteryFeatureMapper

class BatteryDataLoader:
    """
    Loads and processes battery datasets from various formats.
    """
    
    def __init__(self):
        self.feature_mapper = BatteryFeatureMapper()
    
    def load_dataset(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load dataset from file.
        
        Args:
            file_path: Path to the data file
            **kwargs: Additional arguments for pandas read functions
            
        Returns:
            Loaded dataframe
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.csv':
            return pd.read_csv(file_path, **kwargs)
        elif file_ext in ['.xlsx', '.xls']:
            return pd.read_excel(file_path, **kwargs)
        elif file_ext == '.parquet':
            return pd.read_parquet(file_path, **kwargs)
        elif file_ext == '.json':
            return pd.read_json(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def process_dataset(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Process dataset and extract standardized features.
        
        Args:
            file_path: Path to the data file
            **kwargs: Additional arguments for data loading
            
        Returns:
            Dictionary containing processed data and metadata
        """
        # Load raw data
        raw_df = self.load_dataset(file_path, **kwargs)
        
        print(f"Original dataset shape: {raw_df.shape}")
        print(f"Original columns: {list(raw_df.columns)}")
        
        # Find feature mapping
        feature_mapping = self.feature_mapper.find_matching_columns(raw_df.columns.tolist())
        print(f"Detected feature mapping: {feature_mapping}")
        
        # Extract standardized features
        processed_df = self.feature_mapper.extract_features(raw_df, feature_mapping)
        
        print(f"Processed dataset shape: {processed_df.shape}")
        print(f"Available features: {list(processed_df.columns)}")
        
        return {
            'raw_data': raw_df,
            'processed_data': processed_df,
            'feature_mapping': feature_mapping,
            'available_features': list(processed_df.columns)
        }
    
    def process_dataset_from_df(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Process dataset from an existing DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing processed data and metadata
        """
        print(f"Original dataset shape: {df.shape}")
        print(f"Original columns: {list(df.columns)}")
        
        # Find feature mapping
        feature_mapping = self.feature_mapper.find_matching_columns(df.columns.tolist())
        print(f"Detected feature mapping: {feature_mapping}")
        
        # Extract standardized features
        processed_df = self.feature_mapper.extract_features(df, feature_mapping)
        
        print(f"Processed dataset shape: {processed_df.shape}")
        print(f"Available features: {list(processed_df.columns)}")
        
        return {
            'raw_data': df,
            'processed_data': processed_df,
            'feature_mapping': feature_mapping,
            'available_features': list(processed_df.columns)
        }