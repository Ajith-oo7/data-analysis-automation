#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main execution script for AI Data Analysis Automation.
This script orchestrates the end-to-end data analysis process.
"""

import os
import json
import logging
import argparse
import sys
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
import traceback

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), 'config', '.env'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'main.log'), mode='a')
    ]
)

# Set up logger
logger = logging.getLogger(__name__)

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Import modules
from data_ingestion.file_handler import load_file, FileWatcher
from data_processing.data_processor import DataProcessor
from ai_insights.insight_generator import InsightGenerator
from reporting.report_generator import ReportGenerator
from config.config_loader import load_config
# Import forecasting module
from forecasting.forecast_engine import ForecastEngine

# Try importing Snowflake connector
try:
    from data_sources.snowflake_connector import SnowflakeConnector
    HAS_SNOWFLAKE = True
except ImportError:
    logger.warning("Snowflake connector not available. Snowflake data sources will not be available.")
    HAS_SNOWFLAKE = False

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AI Data Analysis Automation - Process data files with AI-powered insights"
    )
    
    parser.add_argument(
        "--data-source", 
        type=str, 
        required=True,
        help="Path to the data file to analyze or Snowflake table (snowflake:table_name)"
    )
    
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["full", "process_only", "insights_only", "forecast_only"],
        default="full",
        help="Processing mode: full (default), process_only, insights_only, or forecast_only"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config.json",
        help="Path to configuration file (default: config/config.json)"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=None,
        help="Custom output directory (overrides config)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    # Forecasting options
    forecast_group = parser.add_argument_group('Forecasting Options')
    
    forecast_group.add_argument(
        "--forecast", 
        action="store_true",
        help="Enable time series forecasting"
    )
    
    forecast_group.add_argument(
        "--date-column", 
        type=str, 
        help="Name of the date column for time series forecasting"
    )
    
    forecast_group.add_argument(
        "--target-column", 
        type=str,
        help="Name of the target column to forecast"
    )
    
    forecast_group.add_argument(
        "--forecast-periods", 
        type=int,
        help="Number of periods to forecast"
    )
    
    # Email notification options
    email_group = parser.add_argument_group('Email Notification Options')
    
    email_group.add_argument(
        "--email", 
        action="store_true",
        help="Enable email notifications"
    )
    
    email_group.add_argument(
        "--email-recipients", 
        type=str,
        help="Comma-separated list of email recipients"
    )
    
    email_group.add_argument(
        "--email-server", 
        type=str,
        help="SMTP server address"
    )
    
    email_group.add_argument(
        "--email-port", 
        type=int,
        help="SMTP server port"
    )
    
    email_group.add_argument(
        "--email-user", 
        type=str,
        help="SMTP username"
    )
    
    email_group.add_argument(
        "--email-password", 
        type=str,
        help="SMTP password"
    )
    
    # Snowflake options
    snowflake_group = parser.add_argument_group('Snowflake Options')
    
    snowflake_group.add_argument(
        "--snowflake-query", 
        type=str,
        help="Custom SQL query to run on Snowflake instead of a table name"
    )
    
    snowflake_group.add_argument(
        "--snowflake-limit", 
        type=int,
        default=10000,
        help="Limit the number of rows to fetch from Snowflake (default: 10000)"
    )
    
    snowflake_group.add_argument(
        "--snowflake-save-local",
        action="store_true", 
        help="Save Snowflake query results to a local CSV file"
    )
    
    return parser.parse_args()

def validate_inputs(args) -> bool:
    """
    Validate command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        True if validation passes, False otherwise
    """
    # Check if data source exists (if it's not a Snowflake reference)
    if not args.data_source.startswith("snowflake:") and not os.path.exists(args.data_source):
        logger.error(f"Data source file not found: {args.data_source}")
        return False
    
    # Check if configuration file exists
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        return False
    
    # If Snowflake reference, check if the connector is available
    if args.data_source.startswith("snowflake:") and not HAS_SNOWFLAKE:
        logger.error("Snowflake connector not available. Please install with: pip install snowflake-connector-python")
        return False
    
    # Forecast validation
    if args.forecast or args.mode == "forecast_only":
        if not args.date_column or not args.target_column:
            logger.error("Date column and target column must be specified for forecasting")
            return False
    
    return True

def process_data(file_path: str, mode: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process data through the analysis pipeline.
    
    Args:
        file_path: Path to the data file or Snowflake table reference
        mode: Processing mode (full, process_only, insights_only, forecast_only)
        config: Configuration dictionary
        
    Returns:
        Dictionary with processing results
    """
    logger.info(f"Starting data processing for {file_path} in {mode} mode")
    result = {
        "status": "success",
        "file_path": file_path,
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": None,
        "output_paths": {}
    }
    
    try:
        # Step 1: Load file
        logger.info("Loading and validating file...")
        file_info = load_file(file_path, config)
        
        if file_info["status"] == "error":
            result["status"] = "error"
            result["message"] = file_info["message"]
            logger.error(f"Error loading file: {file_info['message']}")
            return result
        
        data = file_info["data"]
        result["file_type"] = file_info["file_type"]
        logger.info(f"Successfully loaded file of type: {file_info['file_type']}")
        
        # Step 2: Process data
        if mode in ["full", "process_only", "forecast_only"]:
            data_processor = DataProcessor(config)
            logger.info("Processing data...")
            processed_data = data_processor.process(data, file_path)
            
            if processed_data["status"] == "error":
                result["status"] = "error"
                result["message"] = processed_data["message"]
                logger.error(f"Error processing data: {processed_data['message']}")
                return result
            
            data = processed_data["data"]
            result["output_paths"]["processed_data"] = processed_data["output_path"]
            logger.info(f"Successfully processed data: {processed_data['output_path']}")
        
        # Step 2.5: Generate forecasts if enabled
        if mode in ["full", "forecast_only"] and "forecasting" in config:
            forecast_config = config["forecasting"]
            if forecast_config.get("enabled", False):
                logger.info("Generating time series forecasts...")
                
                # Get forecast parameters
                date_column = forecast_config.get("date_column")
                target_column = forecast_config.get("target_column")
                periods = forecast_config.get("periods", 12)
                
                # Create the forecasting engine
                forecast_engine = ForecastEngine(config)
                
                # Generate forecast
                forecast_result = forecast_engine.generate_forecast(
                    data, 
                    date_column=date_column,
                    target_column=target_column,
                    periods=periods
                )
                
                if forecast_result["status"] == "success":
                    logger.info(f"Successfully generated forecast: {forecast_result['output_path']}")
                    result["output_paths"]["forecast"] = forecast_result["output_path"]
                    
                    # Attach forecast to report generator
                    if "forecast_chart" in forecast_result:
                        result["forecast_chart"] = forecast_result["forecast_chart"]
                else:
                    logger.warning(f"Forecast generation warning: {forecast_result['message']}")
        
        # Step 3: Generate insights
        if mode in ["full", "insights_only"]:
            logger.info("Generating insights...")
            insight_generator = InsightGenerator(config)
            insights_result = insight_generator.generate_insights(data, file_path)
            
            if insights_result["status"] == "error":
                result["status"] = "error"
                result["message"] = insights_result["message"]
                logger.error(f"Error generating insights: {insights_result['message']}")
                return result
            
            result["output_paths"]["insights"] = insights_result["output_path"]
            logger.info(f"Successfully generated insights: {insights_result['output_path']}")
            
            # Step 4: Generate report
            if "report" in insights_result and insights_result["report"]:
                logger.info("Generating report...")
                report_generator = ReportGenerator(config)
                
                # If we have a forecast chart, add it to the report
                if "forecast_chart" in result:
                    report_generator.add_chart("forecast", result["forecast_chart"])
                
                report_result = report_generator.generate_report(
                    data=data,
                    insights_path=insights_result["output_path"],
                    file_path=file_path
                )
                
                if report_result["status"] == "error":
                    result["status"] = "error"
                    result["message"] = report_result["message"]
                    logger.error(f"Error generating report: {report_result['message']}")
                    return result
                
                result["output_paths"]["report"] = report_result["report_paths"]
                logger.info(f"Successfully generated report: {report_result['report_paths']}")
        
        # Update end time
        result["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Data processing completed successfully for {file_path}")
        return result
        
    except Exception as e:
        result["status"] = "error"
        result["message"] = f"Unexpected error: {str(e)}"
        result["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.error(f"Unexpected error during processing: {str(e)}")
        traceback.print_exc()
        return result

def main():
    """Main entry point for the CLI application."""
    # Create necessary directories
    for directory in ['source_data', 'processed_data', 'insights', 'reports', 'logs', 'forecasts']:
        os.makedirs(directory, exist_ok=True)
    
    # Parse arguments
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Validate inputs
    if not validate_inputs(args):
        return 1
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.debug("Configuration loaded successfully")
        
        # Override config with command line arguments
        if args.output_dir:
            config["data_processing"]["output_dir"] = os.path.join(args.output_dir, "processed_data")
            config["ai_insights"]["output_dir"] = os.path.join(args.output_dir, "insights")
            config["reporting"]["output_dir"] = os.path.join(args.output_dir, "reports")
            if "forecasting" in config:
                config["forecasting"]["output_dir"] = os.path.join(args.output_dir, "forecasts")
            logger.info(f"Output directories overridden to use base: {args.output_dir}")
        
        # Configure forecasting if requested
        if args.forecast or args.mode == "forecast_only":
            if "forecasting" not in config:
                config["forecasting"] = {}
            
            config["forecasting"]["enabled"] = True
            
            if args.date_column:
                config["forecasting"]["date_column"] = args.date_column
            
            if args.target_column:
                config["forecasting"]["target_column"] = args.target_column
            
            if args.forecast_periods:
                config["forecasting"]["periods"] = args.forecast_periods
            
            logger.info("Forecasting enabled with parameters: " + 
                      f"date_column={config['forecasting'].get('date_column')}, " +
                      f"target_column={config['forecasting'].get('target_column')}, " +
                      f"periods={config['forecasting'].get('periods', 12)}")
                      
        # Configure email notifications if requested
        if args.email:
            if "reporting" not in config:
                config["reporting"] = {}
            if "email" not in config["reporting"]:
                config["reporting"]["email"] = {}
                
            config["reporting"]["email"]["enabled"] = True
            config["reporting"]["email"]["send_on_completion"] = True
            
            if args.email_recipients:
                config["reporting"]["email"]["recipients"] = args.email_recipients.split(",")
            
            if args.email_server:
                config["reporting"]["email"]["smtp_server"] = args.email_server
            
            if args.email_port:
                config["reporting"]["email"]["smtp_port"] = args.email_port
            
            if args.email_user:
                config["reporting"]["email"]["username"] = args.email_user
            
            if args.email_password:
                config["reporting"]["email"]["password"] = args.email_password
                
            logger.info("Email notifications enabled")
            
        # Configure Snowflake if needed
        if args.data_source.startswith("snowflake:"):
            if "data_sources" not in config:
                config["data_sources"] = {}
            if "snowflake" not in config["data_sources"]:
                config["data_sources"]["snowflake"] = {}
                
            # Check for custom query
            if args.snowflake_query:
                config["data_sources"]["snowflake"]["query"] = args.snowflake_query
                logger.info(f"Using custom Snowflake query: {args.snowflake_query}")
                
            # Set row limit
            if args.snowflake_limit:
                config["data_sources"]["snowflake"]["limit"] = args.snowflake_limit
                logger.info(f"Snowflake row limit set to: {args.snowflake_limit}")
                
            # Check if we should save to local file
            if args.snowflake_save_local:
                config["data_sources"]["snowflake"]["save_local"] = True
                logger.info("Snowflake results will be saved to a local file")
                
        # Process data
        result = process_data(args.data_source, args.mode, config)
        
        if result["status"] == "success":
            logger.info("Processing completed successfully")
            logger.info(f"Processing time: {result['start_time']} to {result['end_time']}")
            
            # Print output paths
            for output_type, output_path in result["output_paths"].items():
                if isinstance(output_path, list):
                    for path in output_path:
                        logger.info(f"{output_type.capitalize()} output: {path}")
                else:
                    logger.info(f"{output_type.capitalize()} output: {output_path}")
            
            print("\nProcessing completed successfully!")
            return 0
        else:
            logger.error(f"Processing failed: {result['message']}")
            print(f"\nProcessing failed: {result['message']}")
            return 1
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        traceback.print_exc()
        print(f"\nUnexpected error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
