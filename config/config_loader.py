import os
import json
import logging
from typing import Dict, Any

# Configure logging
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    logger.info(f"Loading configuration from {config_path}")
    
    try:
        # Check if file exists
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load JSON configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        logger.info("Configuration loaded successfully")
        return config
    
    except FileNotFoundError as e:
        logger.error(str(e))
        raise
    
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing configuration file: {str(e)}")
        raise
    
    except Exception as e:
        logger.error(f"Unexpected error loading configuration: {str(e)}")
        raise

# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    try:
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        config = load_config(config_path)
        print("Configuration loaded successfully:")
        print(json.dumps(config, indent=2))
    except Exception as e:
        print(f"Error loading configuration: {str(e)}") 