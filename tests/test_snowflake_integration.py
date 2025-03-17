import os
import sys
import logging
import pandas as pd
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    # Import components
    from data_sources.snowflake_connector import SnowflakeConnector
    import processing.data_processor as dp
    from insights.insight_generator import InsightGenerator
    from reporting.report_generator import ReportGenerator
    
    HAS_COMPONENTS = True
except ImportError as e:
    logger.error(f"Failed to import components: {e}")
    HAS_COMPONENTS = False

def load_config():
    """Load the configuration for testing."""
    # Default test configuration
    config = {
        "data_sources": {
            "snowflake": {
                # Will use environment variables
            }
        },
        "processing": {
            "clean_missing_values": True,
            "outlier_detection": True,
            "normalization": True
        },
        "insights": {
            "max_insights_per_category": 5
        },
        "reporting": {
            "output_dir": "reports",
            "generate_pdf": False,
            "email": {
                "enabled": False
            }
        }
    }
    
    return config

def test_snowflake_to_report_pipeline():
    """Test the full data pipeline from Snowflake to report generation."""
    if not HAS_COMPONENTS:
        logger.error("Required components not available. Test aborted.")
        return False
    
    try:
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded")
        
        # Initialize Snowflake connector
        snowflake = SnowflakeConnector(config)
        logger.info("Snowflake connector initialized")
        
        # Test connection
        if not snowflake.test_connection():
            logger.error("Cannot connect to Snowflake. Test aborted.")
            return False
        
        # List available tables
        tables = snowflake.list_tables()
        if not tables:
            logger.warning("No tables found in Snowflake schema. Test aborted.")
            return False
        
        # Select the first table for testing
        test_table = tables[0]
        logger.info(f"Selected test table: {test_table}")
        
        # Load data from the table
        df = snowflake.load_table(test_table, limit=1000)  # Limit to 1000 rows for testing
        if df is None or df.empty:
            logger.error(f"Failed to load data from {test_table}. Test aborted.")
            return False
        
        logger.info(f"Loaded {len(df)} rows from {test_table}")
        
        # Process the data
        processor = dp.DataProcessor(config)
        processed_df = processor.process(df)
        logger.info(f"Data processed. Shape: {processed_df.shape}")
        
        # Generate insights
        insights_generator = InsightGenerator(config)
        insights = insights_generator.generate_insights(processed_df)
        
        # Create a temporary file for insights
        import json
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            json.dump(insights, f)
            insights_path = f.name
        
        logger.info(f"Insights generated and saved to {insights_path}")
        
        # Generate report
        report_generator = ReportGenerator(config)
        report_result = report_generator.generate_report(
            data=processed_df,
            insights_path=insights_path,
            file_path=f"snowflake:{test_table}"
        )
        
        # Clean up temporary file
        try:
            os.unlink(insights_path)
        except:
            pass
        
        if report_result["status"] == "success":
            logger.info("Report generation successful")
            for path in report_result["report_paths"]:
                logger.info(f"Report generated: {path}")
            return True
        else:
            logger.error(f"Report generation failed: {report_result['message']}")
            return False
        
    except Exception as e:
        logger.error(f"Error in Snowflake integration test: {str(e)}")
        return False

if __name__ == "__main__":
    print("\n=== Snowflake Integration Test ===\n")
    
    # Check environment setup
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', '.env'))
    missing = []
    for var in ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD", 
                "SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_DATABASE", "SNOWFLAKE_SCHEMA"]:
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        print(f"\n❌ Missing environment variables: {', '.join(missing)}")
        print("Please set these variables in your config/.env file")
        sys.exit(1)
    
    if not HAS_COMPONENTS:
        print("\n❌ Required components not available.")
        print("Please ensure all project modules are properly installed.")
        sys.exit(1)
    
    # Run the test
    success = test_snowflake_to_report_pipeline()
    
    if success:
        print("\n✅ Snowflake integration test passed!")
        print("The complete data pipeline from Snowflake to report generation works correctly.")
    else:
        print("\n❌ Snowflake integration test failed.")
        print("Please check the logs for details on what went wrong.") 