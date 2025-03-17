import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Union, List

# Configure logging
logger = logging.getLogger(__name__)

def clean_data(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Clean the input DataFrame by handling missing values, removing duplicates,
    and converting data types if needed.
    
    Args:
        df: Input DataFrame to clean
        config: Dictionary containing configuration parameters
        
    Returns:
        Cleaned DataFrame
    """
    logger.info("Starting data cleaning process")
    cleaned_df = df.copy()
    
    try:
        # Handle missing values based on configuration
        if config.get("handle_missing_values"):
            missing_strategy = config.get("handle_missing_values")
            
            # Get numeric columns
            numeric_cols = cleaned_df.select_dtypes(include=np.number).columns
            
            if missing_strategy == "drop":
                # Drop rows with any missing values
                original_len = len(cleaned_df)
                cleaned_df = cleaned_df.dropna()
                dropped_rows = original_len - len(cleaned_df)
                if dropped_rows > 0:
                    logger.info(f"Dropped {dropped_rows} rows with missing values")
            
            elif missing_strategy == "mean":
                # Fill missing values in numeric columns with column mean
                for col in numeric_cols:
                    if cleaned_df[col].isna().any():
                        mean_val = cleaned_df[col].mean()
                        cleaned_df[col] = cleaned_df[col].fillna(mean_val)
                        logger.info(f"Filled missing values in '{col}' with mean value: {mean_val}")
            
            elif missing_strategy == "median":
                # Fill missing values in numeric columns with column median
                for col in numeric_cols:
                    if cleaned_df[col].isna().any():
                        median_val = cleaned_df[col].median()
                        cleaned_df[col] = cleaned_df[col].fillna(median_val)
                        logger.info(f"Filled missing values in '{col}' with median value: {median_val}")
            
            elif missing_strategy == "mode":
                # Fill missing values with column mode
                for col in cleaned_df.columns:
                    if cleaned_df[col].isna().any():
                        mode_val = cleaned_df[col].mode()[0]
                        cleaned_df[col] = cleaned_df[col].fillna(mode_val)
                        logger.info(f"Filled missing values in '{col}' with mode value")
            
            elif missing_strategy == "zero":
                # Fill missing values in numeric columns with 0
                for col in numeric_cols:
                    if cleaned_df[col].isna().any():
                        cleaned_df[col] = cleaned_df[col].fillna(0)
                        logger.info(f"Filled missing values in '{col}' with 0")
            
            # For non-numeric columns, fill with 'Unknown' or equivalent
            non_numeric_cols = cleaned_df.select_dtypes(exclude=np.number).columns
            for col in non_numeric_cols:
                if cleaned_df[col].isna().any():
                    cleaned_df[col] = cleaned_df[col].fillna("Unknown")
                    logger.info(f"Filled missing values in non-numeric column '{col}' with 'Unknown'")
        
        # Remove duplicate rows if specified
        if config.get("remove_duplicates"):
            original_len = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates()
            dropped_dups = original_len - len(cleaned_df)
            if dropped_dups > 0:
                logger.info(f"Removed {dropped_dups} duplicate rows")
        
        # Perform additional cleaning steps if needed
        
        logger.info("Data cleaning completed successfully")
        return cleaned_df
    
    except Exception as e:
        logger.error(f"Error during data cleaning: {str(e)}")
        # Return original dataframe if cleaning fails
        return df

def transform_data(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Transform the DataFrame by normalizing, scaling, or encoding features.
    
    Args:
        df: Input DataFrame to transform
        config: Dictionary containing configuration parameters
        
    Returns:
        Transformed DataFrame
    """
    logger.info("Starting data transformation process")
    transformed_df = df.copy()
    
    try:
        # Add transformation logic here based on config
        # Examples might include:
        # - Normalizing numeric columns
        # - One-hot encoding categorical variables
        # - Feature engineering
        
        logger.info("Data transformation completed successfully")
        return transformed_df
    
    except Exception as e:
        logger.error(f"Error during data transformation: {str(e)}")
        # Return original dataframe if transformation fails
        return df

class DataProcessor:
    """
    Class to handle the end-to-end data processing workflow.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DataProcessor with configuration.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.output_dir = config.get("data_processing", {}).get("output_dir", "processed_data")
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def process(self, df: pd.DataFrame, file_name: str) -> Dict[str, Any]:
        """
        Process the input DataFrame by cleaning and transforming it.
        
        Args:
            df: Input DataFrame to process
            file_name: Name of the original file for output naming
            
        Returns:
            Dictionary containing processing results
        """
        logger.info(f"Processing data from file: {file_name}")
        result = {
            "status": "success",
            "data": None,
            "output_path": None,
            "message": ""
        }
        
        try:
            # Step 1: Clean the data
            cleaned_df = clean_data(df, self.config.get("data_processing", {}))
            
            # Step 2: Transform the data
            processed_df = transform_data(cleaned_df, self.config.get("data_processing", {}))
            
            # Step 3: Save the processed data
            base_name = os.path.splitext(os.path.basename(file_name))[0]
            output_path = os.path.join(self.output_dir, f"{base_name}_processed.csv")
            processed_df.to_csv(output_path, index=False)
            
            result["data"] = processed_df
            result["output_path"] = output_path
            result["message"] = f"Data processed successfully and saved to {output_path}"
            logger.info(f"Data processing completed: {output_path}")
            
        except Exception as e:
            result["status"] = "error"
            result["message"] = f"Error processing data: {str(e)}"
            logger.error(f"Error processing data: {str(e)}")
        
        return result

# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Sample configuration
    sample_config = {
        "data_processing": {
            "output_dir": "processed_data",
            "perform_data_cleaning": True,
            "handle_missing_values": "mean",
            "remove_duplicates": True
        }
    }
    
    # Create a sample DataFrame
    data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, 4, 5],
        'C': ['a', 'b', 'c', 'd', 'd']
    }
    sample_df = pd.DataFrame(data)
    
    # Process the data
    processor = DataProcessor(sample_config)
    result = processor.process(sample_df, "sample_data.csv")
    
    if result["status"] == "success":
        print("Processing completed successfully!")
        print(result["data"].head())
        print(f"Output saved to: {result['output_path']}")
    else:
        print(f"Processing failed: {result['message']}") 