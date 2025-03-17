# Snowflake Integration Guide

This guide explains how to configure and test the Snowflake integration with the AI Data Analysis Automation platform.

## Prerequisites

1. A Snowflake account with access credentials
2. Python environment with required packages installed
3. Basic understanding of SQL queries

## Configuration

### Step 1: Set up Environment Variables

Configure your Snowflake credentials in the `.env` file located at `config/.env`. This file should contain the following variables:

```
SNOWFLAKE_ACCOUNT=your_account_identifier
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=your_database
SNOWFLAKE_SCHEMA=your_schema
```

Replace the placeholder values with your actual Snowflake credentials.

### Step 2: Install Required Packages

Install the necessary Python packages using pip:

```bash
pip install python-dotenv snowflake-connector-python pandas
```

Alternatively, you can install all required dependencies:

```bash
pip install -r requirements.txt
```

## Testing the Connection

### Option 1: Using the Test Script

Run the provided test script to verify that your Snowflake connection is working:

```bash
python test_snowflake_connection.py
```

Or use the batch file on Windows:

```
test_snowflake.bat
```

The test script will:
1. Verify that the required packages are installed
2. Check your Snowflake credentials
3. Attempt to connect to your Snowflake account
4. List available tables in your configured database and schema
5. Optionally query a sample from a table you specify

### Option 2: Manual Testing

You can also test the connection directly in your Python environment:

```python
import snowflake.connector
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv("config/.env")

# Connect to Snowflake
conn = snowflake.connector.connect(
    account=os.getenv("SNOWFLAKE_ACCOUNT"),
    user=os.getenv("SNOWFLAKE_USER"),
    password=os.getenv("SNOWFLAKE_PASSWORD"),
    warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
    database=os.getenv("SNOWFLAKE_DATABASE"),
    schema=os.getenv("SNOWFLAKE_SCHEMA")
)

# Test query
cursor = conn.cursor()
cursor.execute("SELECT current_version()")
print(cursor.fetchone()[0])

# Close connection
cursor.close()
conn.close()
```

## Troubleshooting

### Common Issues

1. **Import Error**: If you see errors about missing modules, install them using pip:
   ```
   pip install snowflake-connector-python pandas python-dotenv
   ```

2. **Connection Error**: If you can't connect to Snowflake:
   - Verify your account identifier format (it might need your region, e.g., `myaccount.us-east-1`)
   - Check that your username and password are correct
   - Ensure your IP address is allowed in Snowflake network policies

3. **Access Denied**: If you get permission errors:
   - Check that your user has access to the specified warehouse, database, and schema
   - Verify that the warehouse is running and has enough credits

4. **Python Version Compatibility**: For older Python versions, you may need specific versions of the dependencies:
   ```
   pip install snowflake-connector-python==2.7.0 pandas==1.3.5 python-dotenv==0.15.0
   ```

## Using Snowflake in the Application

Once your connection is confirmed working, you can use Snowflake data in your analysis:

### Loading Data from Snowflake

```python
from snowflake_data_connector import SnowflakeConnector

# Initialize connector with config
connector = SnowflakeConnector(config)

# Load data
df = connector.query_data("SELECT * FROM my_table LIMIT 1000")

# Process data with the platform's analysis tools
processor = DataProcessor(config)
results = processor.process(df)
```

### Running Custom Queries

You can also execute custom SQL queries and load the results directly:

```python
custom_query = """
SELECT 
    date_column, 
    SUM(value_column) as total_value
FROM 
    my_table
WHERE 
    category = 'important'
GROUP BY 
    date_column
ORDER BY 
    date_column
"""

results_df = connector.query_data(custom_query)
```

## Next Steps

After successfully configuring and testing your Snowflake connection:

1. Set up automated data refresh in the config file
2. Create custom SQL queries for your specific analytics needs
3. Configure result caching to improve performance
4. Set up report generation for Snowflake data insights

For more information, refer to the Snowflake documentation or contact your database administrator. 