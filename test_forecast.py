#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the forecasting module.
This script demonstrates how to use the ForecastEngine.
"""

import os
import pandas as pd
import json
from forecasting.forecast_engine import ForecastEngine
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Main function to test forecasting capabilities."""
    # Create config
    config = {
        "forecasting": {
            "output_dir": "forecasts",
            "default_model": "sma",  # Using simple moving average for demonstration
            "confidence_interval": 0.95,
            "date_column": "date",
            "target_column": "sales",
            "forecast_periods": 10
        }
    }
    
    # Create forecasting engine
    forecast_engine = ForecastEngine(config)
    
    # Load test data
    data_path = "source_data/sample_timeseries.csv"
    if not os.path.exists(data_path):
        logger.error(f"Test data not found: {data_path}")
        return
    
    data = pd.read_csv(data_path)
    logger.info(f"Loaded test data with {len(data)} rows")
    
    # Ensure date column is datetime
    data["date"] = pd.to_datetime(data["date"])
    
    # Generate forecast
    forecast_result = forecast_engine.forecast(
        data=data,
        date_column="date",
        target_column="sales",
        periods=10,
        model="sma"
    )
    
    # Log forecast results
    logger.info(f"Forecast generated using {forecast_result['model']} model")
    logger.info(f"Forecast chart saved to {forecast_result['chart_path']}")
    
    # Save forecast to file
    os.makedirs("forecasts", exist_ok=True)
    forecast_path = os.path.join("forecasts", "test_forecast.json")
    with open(forecast_path, 'w') as f:
        json.dump(forecast_result, f, indent=2)
    
    logger.info(f"Forecast data saved to {forecast_path}")
    
    # Display forecast values
    print("\nForecast Values:")
    for i, (date, value) in enumerate(zip(forecast_result["forecast"]["dates"], forecast_result["forecast"]["values"])):
        print(f"{date}: {value:.2f}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main() 