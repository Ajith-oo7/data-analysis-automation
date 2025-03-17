import os
import time
import pandas as pd
import json
import logging
from typing import Optional, Dict, Any, Union, Callable, List
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import glob
import threading
import traceback

# Configure logging
logger = logging.getLogger(__name__)

# Try importing Snowflake connector
try:
    from data_sources.snowflake_connector import SnowflakeConnector
    HAS_SNOWFLAKE = True
except ImportError:
    logger.warning("Snowflake connector not available. Install with: pip install snowflake-connector-python")
    HAS_SNOWFLAKE = False

# Define supported file types
SUPPORTED_FILE_TYPES = ['.csv', '.xls', '.xlsx', '.json', '.parquet']

class FileProcessingError(Exception):
    """Custom exception for file processing errors."""
    pass

def detect_file_type(file_path: str) -> str:
    """
    Detect the file type based on extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File extension in lowercase
    """
    _, ext = os.path.splitext(file_path)
    return ext.lower()

def is_file_type_supported(file_path: str) -> bool:
    """
    Check if the file type is supported.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Boolean indicating if file type is supported
    """
    file_type = detect_file_type(file_path)
    return file_type in SUPPORTED_FILE_TYPES

def validate_dataframe_structure(df: pd.DataFrame, required_columns: Optional[list] = None) -> Dict[str, Any]:
    """
    Validate that the DataFrame has the required columns and no null values in critical columns.
    
    Args:
        df: Pandas DataFrame to validate
        required_columns: List of column names that must be present
        
    Returns:
        Dictionary with validation results
    """
    validation_result = {"valid": True, "errors": []}
    
    # Check for empty DataFrame
    if df.empty:
        validation_result["valid"] = False
        validation_result["errors"].append("DataFrame is empty")
        return validation_result
    
    # Check for required columns if specified
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Check for completely null columns
    null_columns = [col for col in df.columns if df[col].isnull().all()]
    if null_columns:
        validation_result["valid"] = False
        validation_result["errors"].append(f"Columns with all null values: {', '.join(null_columns)}")
    
    return validation_result

def load_file(file_path: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load and validate a data file.
    
    Args:
        file_path: Path to the file or Snowflake table identifier
        config: Configuration parameters
        
    Returns:
        Dictionary containing the loaded data and metadata
    """
    result = {
        "status": "error",
        "message": "",
        "data": None,
        "file_type": None
    }
    
    try:
        # Check if it's a Snowflake reference
        if file_path.startswith("snowflake:"):
            return load_from_snowflake(file_path, config)
    
        # Check if file exists
        if not os.path.exists(file_path):
            result["message"] = f"File not found: {file_path}"
            return result
        
        # Get file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Check if file type is supported
        if file_ext not in SUPPORTED_FILE_TYPES:
            result["message"] = f"Unsupported file type: {file_ext}. Supported types: {', '.join(SUPPORTED_FILE_TYPES)}"
            return result
        
        # Load file based on extension
        if file_ext == ".csv":
            data = pd.read_csv(file_path)
        elif file_ext in [".xlsx", ".xls"]:
            data = pd.read_excel(file_path)
        elif file_ext == ".json":
            with open(file_path, 'r') as f:
                json_data = json.load(f)
            
            # Try to convert JSON to DataFrame
            if isinstance(json_data, list):
                data = pd.DataFrame(json_data)
            else:
                result["message"] = "JSON file must contain a list of records"
                return result
        elif file_ext == ".parquet":
            data = pd.read_parquet(file_path)
                
        # Basic validation
        if data is None or len(data) == 0:
            result["message"] = "File is empty"
            return result
            
        # Update result
        result["status"] = "success"
        result["data"] = data
        result["file_type"] = file_ext
        result["message"] = f"Successfully loaded file: {file_path}"
        
        logger.info(f"Successfully loaded file: {file_path}")
        
        return result
    
    except Exception as e:
        result["message"] = f"Error loading file: {str(e)}"
        logger.error(f"Error loading file {file_path}: {str(e)}")
        return result

def load_from_snowflake(table_path: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load data from a Snowflake table.
    
    Args:
        table_path: Snowflake table identifier (format: snowflake:table_name)
        config: Configuration dictionary
        
    Returns:
        Dictionary containing the loaded data and metadata
    """
    result = {
        "status": "error",
        "message": "",
        "data": None,
        "file_type": "snowflake"
    }
    
    if not HAS_SNOWFLAKE:
        result["message"] = "Snowflake connector not available. Install with: pip install snowflake-connector-python"
        return result
    
    try:
        # Extract table name from path
        table_name = table_path.replace("snowflake:", "").strip()
        
        if not table_name:
            result["message"] = "Invalid Snowflake table name"
            return result
        
        # Initialize Snowflake connector
        snowflake = SnowflakeConnector(config or {})
        
        # Load data from table
        logger.info(f"Loading data from Snowflake table: {table_name}")
        data = snowflake.load_table(table_name)
        
        if data is None:
            result["message"] = f"Failed to load data from Snowflake table: {table_name}"
            return result
        
        if len(data) == 0:
            result["message"] = f"Snowflake table is empty: {table_name}"
            return result
        
        # Update result
        result["status"] = "success"
        result["data"] = data
        result["message"] = f"Successfully loaded data from Snowflake table: {table_name}"
        
        logger.info(f"Successfully loaded {len(data)} rows from Snowflake table: {table_name}")
        
        return result
        
    except Exception as e:
        result["message"] = f"Error loading data from Snowflake: {str(e)}"
        logger.error(f"Error loading data from Snowflake: {str(e)}")
        traceback.print_exc()
        return result

class FileWatcher:
    """
    Watch a directory for new files and process them as they appear.
    """
    def __init__(self, directory: str, callback: Callable[[str], None], file_pattern: str = "*.*"):
        """
        Initialize the FileWatcher.
        
        Args:
            directory: Directory to watch
            callback: Function to call when a new file is detected
            file_pattern: File pattern to match (default: *.*) 
        """
        self.directory = directory
        self.callback = callback
        self.file_pattern = file_pattern
        self.is_running = False
        self.processed_files = set()
        self.watch_thread = None
        
    def _get_files(self) -> List[str]:
        """
        Get all files in the watched directory that match the pattern.
        
        Returns:
            List of file paths
        """
        return glob.glob(os.path.join(self.directory, self.file_pattern))
    
    def _watch_directory(self):
        """Watch directory for new files."""
        logger.info(f"Starting to watch directory: {self.directory}")
        
        # Process existing files first
        existing_files = self._get_files()
        for file_path in existing_files:
            self.processed_files.add(file_path)
        
        # Watch for new files
        while self.is_running:
            try:
                current_files = self._get_files()
                
                # Find new files
                for file_path in current_files:
                    if file_path not in self.processed_files:
                        logger.info(f"New file detected: {file_path}")
                        self.processed_files.add(file_path)
                        
                        try:
                            self.callback(file_path)
                        except Exception as e:
                            logger.error(f"Error processing file {file_path}: {str(e)}")
                
                # Sleep to avoid high CPU usage
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in file watcher: {str(e)}")
                time.sleep(10)  # Sleep longer on error
    
    def start(self):
        """Start watching the directory."""
        if self.is_running:
            logger.warning("File watcher is already running")
            return
        
        self.is_running = True
        self.watch_thread = threading.Thread(target=self._watch_directory)
        self.watch_thread.daemon = True
        self.watch_thread.start()
        
    def stop(self):
        """Stop watching the directory."""
        if not self.is_running:
            logger.warning("File watcher is not running")
            return
        
        self.is_running = False
        if self.watch_thread:
            self.watch_thread.join(timeout=1.0)
            logger.info("File watcher stopped")

# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example function to process files
    def process_file(file_path):
        result = load_file(file_path)
        if result["status"] == "success":
            print(f"Successfully processed {file_path}")
            print(f"DataFrame shape: {result['data'].shape}")
            print(result['data'].head())
        else:
            print(f"Failed to process {file_path}: {result['message']}")
    
    # Create the source_data directory if running as main script
    source_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "source_data")
    os.makedirs(source_dir, exist_ok=True)
    
    # Example: Process a single file
    file_path = os.path.join(source_dir, "sample.csv")
    if os.path.exists(file_path):
        result = load_file(file_path)
        if result["status"] == "success":
            print("File Loaded Successfully:")
            print(result["data"].head())
        else:
            print(f"Error: {result['message']}")
    
    # Example: Start watching a directory for new files
    print(f"Watching directory {source_dir} for new files. Press Ctrl+C to stop.")
    watcher = FileWatcher(source_dir, process_file)
    watcher.start() 