import os
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check Python version
python_version = sys.version_info
logger.info(f"Python version: {sys.version}")

try:
    # Import required modules
    from dotenv import load_dotenv
    logger.info("Successfully imported dotenv")
    
    # Try importing snowflake
    try:
        import snowflake.connector
        logger.info("Successfully imported snowflake.connector")
        
        # Try importing pandas
        try:
            import pandas as pd
            logger.info("Successfully imported pandas")
            HAS_PANDAS = True
        except ImportError:
            logger.warning("Failed to import pandas - limited functionality available")
            HAS_PANDAS = False
            
        HAS_SNOWFLAKE = True
    except ImportError:
        logger.error("Failed to import snowflake.connector - please install with: pip install snowflake-connector-python")
        HAS_SNOWFLAKE = False
    
except ImportError:
    logger.error("Failed to import dotenv - please install with: pip install python-dotenv")
    sys.exit(1)

def test_snowflake_connection():
    """Test the Snowflake connection using the configured credentials."""
    
    if not HAS_SNOWFLAKE:
        logger.error("Snowflake connector not available. Please install it first.")
        return False
    
    # Load environment variables from .env file
    env_path = os.path.join('config', '.env')
    if not os.path.exists(env_path):
        logger.error(f"Environment file not found at {env_path}")
        return False
        
    load_dotenv(env_path)
    
    # Get Snowflake credentials from environment variables
    account = os.getenv('SNOWFLAKE_ACCOUNT')
    user = os.getenv('SNOWFLAKE_USER')
    password = os.getenv('SNOWFLAKE_PASSWORD')
    warehouse = os.getenv('SNOWFLAKE_WAREHOUSE')
    database = os.getenv('SNOWFLAKE_DATABASE')
    schema = os.getenv('SNOWFLAKE_SCHEMA')
    
    # Check if all required credentials are provided
    missing = []
    if not account: missing.append('SNOWFLAKE_ACCOUNT')
    if not user: missing.append('SNOWFLAKE_USER')
    if not password: missing.append('SNOWFLAKE_PASSWORD')
    if not warehouse: missing.append('SNOWFLAKE_WAREHOUSE')
    if not database: missing.append('SNOWFLAKE_DATABASE')
    if not schema: missing.append('SNOWFLAKE_SCHEMA')
    
    if missing:
        logger.error(f"Missing Snowflake credentials: {', '.join(missing)}")
        return False
    
    try:
        logger.info("Attempting to connect to Snowflake...")
        
        # Create a connection to Snowflake
        conn = snowflake.connector.connect(
            account=account,
            user=user,
            password=password,
            warehouse=warehouse,
            database=database,
            schema=schema
        )
        
        # Execute a simple query to test the connection
        cursor = conn.cursor()
        cursor.execute("SELECT current_version()")
        version = cursor.fetchone()[0]
        logger.info(f"Successfully connected to Snowflake. Version: {version}")
        
        # List available tables
        logger.info("Retrieving available tables...")
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        
        if tables:
            logger.info(f"Found {len(tables)} tables in {database}.{schema}:")
            for table in tables:
                logger.info(f"  - {table[1]}")
        else:
            logger.info(f"No tables found in {database}.{schema}")
        
        # Close the connection
        cursor.close()
        conn.close()
        logger.info("Connection closed successfully")
        
        return True
    
    except Exception as e:
        logger.error(f"Error connecting to Snowflake: {str(e)}")
        return False

def query_sample_data(table_name, limit=10):
    """Query a sample of data from a specified table."""
    
    if not HAS_SNOWFLAKE:
        logger.error("Snowflake connector not available. Please install it first.")
        return None
        
    if not HAS_PANDAS:
        logger.error("Pandas not available. Please install it first.")
        return None
    
    # Load environment variables from .env file
    load_dotenv(os.path.join('config', '.env'))
    
    # Get Snowflake credentials from environment variables
    account = os.getenv('SNOWFLAKE_ACCOUNT')
    user = os.getenv('SNOWFLAKE_USER')
    password = os.getenv('SNOWFLAKE_PASSWORD')
    warehouse = os.getenv('SNOWFLAKE_WAREHOUSE')
    database = os.getenv('SNOWFLAKE_DATABASE')
    schema = os.getenv('SNOWFLAKE_SCHEMA')
    
    try:
        # Create a connection to Snowflake
        conn = snowflake.connector.connect(
            account=account,
            user=user,
            password=password,
            warehouse=warehouse,
            database=database,
            schema=schema
        )
        
        # Query sample data
        logger.info(f"Querying sample data from {table_name}...")
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        
        # Use pandas to query and return as DataFrame
        df = pd.read_sql(query, conn)
        
        # Close the connection
        conn.close()
        
        logger.info(f"Retrieved {len(df)} rows from {table_name}")
        return df
    
    except Exception as e:
        logger.error(f"Error querying data from {table_name}: {str(e)}")
        return pd.DataFrame() if HAS_PANDAS else None

if __name__ == "__main__":
    print("\n=== Snowflake Connection Tester ===\n")
    
    if not HAS_SNOWFLAKE:
        print("\n⚠️ Snowflake connector not installed. Please run:")
        print("pip install snowflake-connector-python")
        sys.exit(1)
    
    # Test the Snowflake connection
    connection_successful = test_snowflake_connection()
    
    if connection_successful:
        print("\n✅ Snowflake connection test successful!")
        
        if HAS_PANDAS:
            # Prompt for table to query
            try:
                table_input = input("\nEnter a table name to query (or press Enter to skip): ")
            except NameError:  # Python 2 compatibility
                table_input = raw_input("\nEnter a table name to query (or press Enter to skip): ")
            
            if table_input.strip():
                # Query sample data
                df = query_sample_data(table_input)
                
                if df is not None and not df.empty:
                    print("\nSample data:")
                    print(df.head())
                    
                    # Save to CSV for inspection
                    output_file = f"{table_input}_sample.csv"
                    df.to_csv(output_file, index=False)
                    print(f"\nSample data saved to {output_file}")
        else:
            print("\n⚠️ Pandas not installed. Skipping data sample query.")
            print("To enable this feature, run: pip install pandas")
    else:
        print("\n❌ Snowflake connection test failed. Please check your configuration.") 