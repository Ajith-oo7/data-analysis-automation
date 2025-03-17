"""
Data source connectors for the AI Data Analysis Automation platform.

This package contains modules for connecting to various data sources:
- File-based data (CSV, Excel, Parquet)
- Database connections (SQL, NoSQL)
- Cloud storage (S3, Azure, GCP)
- APIs and web services
- Snowflake data warehouse
"""

# Import connectors for easy access
try:
    from .snowflake_connector import SnowflakeConnector
except ImportError:
    pass 