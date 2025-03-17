# AI Data Analysis Automation Platform

A comprehensive platform for automated data analysis, insights generation, time series forecasting, and report generation, now with Snowflake integration.

## Features

- **Data Ingestion**: Support for multiple data sources including CSV, Excel, JSON, Parquet, and Snowflake
- **Data Processing**: Automated data cleaning, transformation, and preparation
- **AI Insights**: Generate automated insights from data using machine learning
- **Time Series Forecasting**: Predict future values based on historical data patterns
- **Report Generation**: Create professional HTML and PDF reports with visualizations
- **Email Notifications**: Send reports and notifications via email
- **Snowflake Integration**: Connect directly to Snowflake data warehouse

## Installation

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

The platform is configured using the `config/config.json` file. Create a `.env` file in the `config` directory for storing sensitive credentials.

### Snowflake Configuration

Add your Snowflake credentials to the `.env` file:

```
SNOWFLAKE_ACCOUNT=your_account_identifier
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=your_database
SNOWFLAKE_SCHEMA=your_schema
```

## Usage

### Basic Usage

```bash
python main.py --data-source path/to/your/data.csv --mode full
```

### Using Snowflake Data Sources

```bash
python main.py --data-source snowflake:your_table_name --mode full
```

### Using a Custom Snowflake Query

```bash
python main.py --data-source snowflake:results --snowflake-query "SELECT * FROM sales WHERE region='Europe' LIMIT 1000" --mode full
```

### Enabling Forecasting

```bash
python main.py --data-source snowflake:sales_data --forecast --date-column order_date --target-column revenue --forecast-periods 30
```

### Sending Email Notifications

```bash
python main.py --data-source snowflake:customer_data --email --email-recipients user@example.com
```

## Testing Snowflake Connection

To test your Snowflake connection:

```bash
python test_snowflake_connection.py
```

Or use the batch file (Windows):

```
test_snowflake.bat
```

## Documentation

Refer to the `docs` directory for detailed documentation:
- [Snowflake Integration Guide](docs/snowflake_integration.md)
- [Time Series Forecasting Guide](docs/forecasting.md)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Sharing the Project (Git)

This project is set up to be safely shared via Git, with sensitive information excluded.

### Setting Up Git for This Project

1. Initialize Git in the project directory (if not already done):
   ```bash
   git init
   ```

2. The `.gitignore` file is already configured to exclude sensitive files like `.env`

3. Stage and commit your files:
   ```bash
   git add .
   git commit -m "Initial commit"
   ```

4. Add your remote repository:
   ```bash
   git remote add origin <your-git-repository-url>
   ```

5. Push to the remote repository:
   ```bash
   git push -u origin main
   ```

### Note for Collaborators

When cloning this repository, you'll need to:

1. Create your own `.env` file in the `config` directory
2. Use the `.env.template` file as a reference for required variables
3. Add your own credentials and API keys to the `.env` file

Never commit the `.env` file with real credentials to the repository.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
