# Time Series Forecasting

This document describes the time series forecasting capabilities added to the AI Data Analysis Automation platform.

## Overview

The forecasting module provides the ability to analyze time series data and generate predictions for future values. It uses statistical models to identify patterns in historical data and extrapolate those patterns into the future.

## Features

- **Multiple Forecasting Models**: 
  - ARIMA (AutoRegressive Integrated Moving Average) for complex time series
  - SMA (Simple Moving Average) for basic forecasting

- **Confidence Intervals**: All forecasts include confidence intervals to show the range of possible future values

- **Automatic Model Parameter Selection**: The system automatically selects appropriate parameters for the models

- **Visualization**: Forecasts are visualized with clear charts showing historical data, forecast values, and confidence intervals

- **Forecast Integration**: Forecasts are fully integrated with the reporting system to include them in HTML reports

## Usage

### Command Line Arguments

The following command-line arguments have been added to control forecasting:

```
--forecast              Enable time series forecasting
--date-column NAME      Name of the column containing dates
--target-column NAME    Name of the column to forecast
--forecast-periods N    Number of periods to forecast
--forecast-model MODEL  Model to use (arima, sma)
```

### Example Command

```
python main.py --data-source source_data/sample_timeseries.csv --mode full --forecast --date-column date --target-column sales --forecast-periods 10
```

### Configuration

You can also configure forecasting in the config.json file:

```json
{
  "forecasting": {
    "output_dir": "forecasts",
    "default_model": "arima",
    "confidence_interval": 0.95,
    "forecast_periods": 30,
    "date_column": "date",
    "target_column": "value",
    "models": {
      "arima": {
        "max_order": [5, 2, 5]
      },
      "sma": {
        "window_size": 0.3
      }
    },
    "visualization": {
      "include_confidence_intervals": true,
      "chart_style": "seaborn"
    }
  }
}
```

## Output

The forecasting module produces:

1. **JSON Forecast Data**: Detailed forecast data including confidence intervals
2. **Chart Image**: Visualization of the forecast
3. **Integration with Reports**: Forecast visualizations are included in HTML reports

## Technical Details

### Required Packages

For ARIMA modeling:
```
pip install statsmodels
```

### ForecastEngine Methods

- `forecast(data, date_column, target_column, periods, model)`: Main method to generate forecasts

### Handling Missing Dependencies

The module gracefully handles missing dependencies by falling back to simpler methods when advanced statistical packages are not available.

## Future Enhancements

Potential future enhancements include:

- Support for seasonal models (SARIMA)
- Machine learning-based forecasting (Prophet, LSTM)
- Automatic feature selection for multivariate forecasting
- Anomaly detection in forecasting
- Forecast accuracy evaluation 