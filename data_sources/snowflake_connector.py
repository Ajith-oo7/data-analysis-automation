import os
import logging
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

# Global flag for Snowflake availability
try:
    import snowflake.connector
    from snowflake.connector.pandas_tools import write_pandas
    HAS_SNOWFLAKE = True
except ImportError:
    logger.warning("Snowflake connector not available. Install with: pip install snowflake-connector-python")
    HAS_SNOWFLAKE = False

class SnowflakeConnector:
    """Connector for Snowflake data sources."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Snowflake connector with configuration.
        
        Args:
            config: Configuration dictionary with Snowflake settings
        """
        self.config = config
        self.snowflake_config = config.get("data_sources", {}).get("snowflake", {})
        
        # Load environment variables for credentials if not in config
        if not self.snowflake_config.get("account"):
            load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', '.env'))
            
            # Get credentials from environment
            self.account = os.getenv("SNOWFLAKE_ACCOUNT")
            self.user = os.getenv("SNOWFLAKE_USER")
            self.password = os.getenv("SNOWFLAKE_PASSWORD")
            self.warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
            self.database = os.getenv("SNOWFLAKE_DATABASE")
            self.schema = os.getenv("SNOWFLAKE_SCHEMA")
        else:
            # Get credentials from config
            self.account = self.snowflake_config.get("account")
            self.user = self.snowflake_config.get("user")
            self.password = self.snowflake_config.get("password")
            self.warehouse = self.snowflake_config.get("warehouse")
            self.database = self.snowflake_config.get("database")
            self.schema = self.snowflake_config.get("schema")
        
        # Verify Snowflake availability
        if not HAS_SNOWFLAKE:
            logger.error("Snowflake connector not available. Install with: pip install snowflake-connector-python")
    
    def _get_connection(self):
        """
        Create a Snowflake connection.
        
        Returns:
            Snowflake connection object or None if connection fails
        """
        if not HAS_SNOWFLAKE:
            logger.error("Snowflake connector not available. Install with: pip install snowflake-connector-python")
            return None
            
        try:
            conn = snowflake.connector.connect(
                account=self.account,
                user=self.user,
                password=self.password,
                warehouse=self.warehouse,
                database=self.database,
                schema=self.schema
            )
            
            return conn
        except Exception as e:
            logger.error(f"Error connecting to Snowflake: {str(e)}")
            return None
    
    def test_connection(self) -> bool:
        """
        Test the Snowflake connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        conn = self._get_connection()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT current_version()")
                version = cursor.fetchone()[0]
                logger.info(f"Successfully connected to Snowflake. Version: {version}")
                cursor.close()
                conn.close()
                return True
            except Exception as e:
                logger.error(f"Error executing test query: {str(e)}")
                return False
        return False
    
    def list_tables(self) -> List[str]:
        """
        List available tables in the configured database and schema.
        
        Returns:
            List of table names
        """
        conn = self._get_connection()
        if not conn:
            return []
            
        try:
            cursor = conn.cursor()
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            
            table_names = [table[1] for table in tables]  # Table name is in the second column
            
            cursor.close()
            conn.close()
            
            return table_names
        except Exception as e:
            logger.error(f"Error listing tables: {str(e)}")
            return []
    
    def query_data(self, query: str) -> Optional[pd.DataFrame]:
        """
        Execute a SQL query and return results as a DataFrame.
        
        Args:
            query: SQL query to execute
            
        Returns:
            Pandas DataFrame with query results or None if query fails
        """
        if not HAS_SNOWFLAKE:
            logger.error("Snowflake connector not available. Install with: pip install snowflake-connector-python")
            return None
            
        conn = self._get_connection()
        if not conn:
            return None
            
        try:
            logger.info(f"Executing query: {query}")
            
            # Execute query and load results into DataFrame
            df = pd.read_sql(query, conn)
            
            conn.close()
            
            logger.info(f"Query returned {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            if conn:
                conn.close()
            return None
    
    def load_table(self, table_name: str, limit: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Load a table from Snowflake into a DataFrame.
        
        Args:
            table_name: Name of the table to load
            limit: Maximum number of rows to load (None for all rows)
            
        Returns:
            Pandas DataFrame with table data or None if loading fails
        """
        limit_clause = f" LIMIT {limit}" if limit else ""
        query = f"SELECT * FROM {table_name}{limit_clause}"
        
        return self.query_data(query)
    
    def execute_query(self, query: str) -> Union[bool, List[Dict]]:
        """
        Execute a SQL query that doesn't return a DataFrame.
        Good for DDL and DML operations.
        
        Args:
            query: SQL query to execute
            
        Returns:
            True if successful, False if failed, or list of results if query returns data
        """
        conn = self._get_connection()
        if not conn:
            return False
            
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            
            # Check if there are results
            try:
                results = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                
                # Convert to list of dictionaries
                result_list = []
                for row in results:
                    result_dict = {}
                    for i, value in enumerate(row):
                        result_dict[columns[i]] = value
                    result_list.append(result_dict)
                
                cursor.close()
                conn.close()
                return result_list
            except:
                # No results, just a success indicator
                conn.commit()  # Commit any changes
                cursor.close()
                conn.close()
                return True
            
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            if conn:
                try:
                    conn.close()
                except:
                    pass  # Ignore closing errors
            return False
    
    def write_dataframe(self, df: pd.DataFrame, table_name: str, mode: str = "replace") -> bool:
        """
        Write a DataFrame to a Snowflake table.
        
        Args:
            df: Pandas DataFrame to write
            table_name: Target table name
            mode: Write mode ('replace' or 'append')
            
        Returns:
            True if successful, False otherwise
        """
        if not HAS_SNOWFLAKE:
            logger.error("Snowflake connector not available. Install with: pip install snowflake-connector-python")
            return False
            
        conn = self._get_connection()
        if not conn:
            return False
            
        try:
            # Create table if mode is replace
            if mode.lower() == "replace":
                cursor = conn.cursor()
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                conn.commit()
                cursor.close()
            
            # Write DataFrame to Snowflake
            success, num_chunks, num_rows, output = write_pandas(
                conn=conn,
                df=df,
                table_name=table_name,
                auto_create_table=True
            )
            
            conn.close()
            
            if success:
                logger.info(f"Successfully wrote {num_rows} rows to {table_name}")
                return True
            else:
                logger.error(f"Failed to write data to {table_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error writing DataFrame to Snowflake: {str(e)}")
            if conn:
                conn.close()
            return False
            
# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Sample configuration
    sample_config = {
        "data_sources": {
            "snowflake": {
                # Leave empty to use environment variables
            }
        }
    }
    
    # Create connector
    connector = SnowflakeConnector(sample_config)
    
    # Test connection
    if connector.test_connection():
        print("Connection successful!")
        
        # List tables
        tables = connector.list_tables()
        print(f"\nAvailable tables ({len(tables)}):")
        for table in tables:
            print(f"- {table}")
        
        # Query sample data if tables exist
        if tables:
            sample_table = tables[0]
            print(f"\nQuerying sample data from {sample_table}...")
            
            df = connector.load_table(sample_table, limit=5)
            if df is not None and not df.empty:
                print("\nSample data:")
                print(df)
    else:
        print("Connection failed. Please check your Snowflake configuration.") 