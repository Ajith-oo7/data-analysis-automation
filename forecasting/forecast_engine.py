"""
Forecast Engine for time series analysis and prediction.

This module implements time series forecasting capabilities using various
statistical models including ARIMA, Exponential Smoothing, and others.
"""

import pandas as pd
import numpy as np
import logging
import os
import json
from typing import Dict, Any, Optional, List, Tuple
import matplotlib.pyplot as plt

# Set non-interactive backend for matplotlib
plt.switch_backend('agg')

logger = logging.getLogger(__name__)

# Optional dependencies for advanced forecasting
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger.warning("statsmodels not installed. Advanced forecasting capabilities will be limited.")

class ForecastEngine:
    """Engine for time series forecasting capabilities."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the forecast engine with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.forecast_config = config.get("forecasting", {})
        self.default_model = self.forecast_config.get("default_model", "arima")
        self.confidence_interval = self.forecast_config.get("confidence_interval", 0.95)
        self.output_dir = self.forecast_config.get("output_dir", "forecasts")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
    def forecast(self, data: pd.DataFrame, 
                 date_column: str,
                 target_column: str,
                 periods: int = 30,
                 model: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate forecasts for time series data.
        
        Args:
            data: DataFrame containing time series data
            date_column: Name of the column containing dates
            target_column: Name of the column to forecast
            periods: Number of periods to forecast
            model: Model to use for forecasting (arima, ets, etc.)
            
        Returns:
            Dictionary containing forecast results
        """
        model = model or self.default_model
        model = model.lower()
        
        # Check if required libraries are available
        if model == "arima" and not STATSMODELS_AVAILABLE:
            logger.warning("statsmodels not available for ARIMA forecasting. Using simple moving average.")
            model = "sma"
            
        logger.info(f"Generating {periods} period forecast for {target_column} using {model} model")
        
        # Prepare time series data
        try:
            # Ensure data is sorted by date
            if date_column not in data.columns:
                raise ValueError(f"Date column '{date_column}' not found in data")
                
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
                
            # Convert date column to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
                data[date_column] = pd.to_datetime(data[date_column])
                
            # Sort by date
            data = data.sort_values(by=date_column)
            
            # Set date as index for time series analysis
            ts_data = data.set_index(date_column)[target_column].copy()
            
            # Initialize result structure
            result = {
                "model": model,
                "target_column": target_column,
                "periods": periods,
                "original_data": {
                    "dates": [str(d) for d in ts_data.index],
                    "values": ts_data.tolist()
                },
                "forecast": {},
                "confidence_intervals": {},
                "model_params": {},
                "performance_metrics": {}
            }
            
            # Generate forecast based on model
            if model == "arima":
                self._generate_arima_forecast(ts_data, periods, result)
            elif model == "sma":
                self._generate_sma_forecast(ts_data, periods, result)
            else:
                logger.warning(f"Unsupported model: {model}. Using simple moving average.")
                self._generate_sma_forecast(ts_data, periods, result)
                
            # Generate visualization
            chart_path = self._generate_forecast_chart(ts_data, result, target_column)
            result["chart_path"] = chart_path
            
            logger.info(f"Forecast generated successfully. Chart saved to {chart_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            return {
                "error": str(e),
                "model": model,
                "target_column": target_column
            }
            
    def _generate_arima_forecast(self, ts_data: pd.Series, periods: int, result: Dict[str, Any]) -> None:
        """
        Generate forecast using ARIMA model.
        
        Args:
            ts_data: Time series data
            periods: Number of periods to forecast
            result: Dictionary to store results (modified in-place)
        """
        if not STATSMODELS_AVAILABLE:
            logger.error("statsmodels not available for ARIMA forecasting")
            result["error"] = "statsmodels not available for ARIMA forecasting"
            return
            
        try:
            # Simple automatic order selection
            p, d, q = self._auto_arima_order(ts_data)
            logger.info(f"Selected ARIMA order: ({p}, {d}, {q})")
            
            # Fit ARIMA model
            model = ARIMA(ts_data, order=(p, d, q))
            model_fit = model.fit()
            
            # Generate forecast
            forecast = model_fit.forecast(steps=periods)
            
            # Extract confidence intervals
            conf_int = model_fit.get_forecast(steps=periods).conf_int(alpha=1-self.confidence_interval)
            
            # Store forecast results
            forecast_dates = pd.date_range(start=ts_data.index[-1], periods=periods+1, freq=self._infer_frequency(ts_data))[1:]
            
            result["forecast"] = {
                "dates": [str(d) for d in forecast_dates],
                "values": forecast.tolist()
            }
            
            result["confidence_intervals"] = {
                "lower": conf_int.iloc[:, 0].tolist(),
                "upper": conf_int.iloc[:, 1].tolist()
            }
            
            # Store model parameters
            result["model_params"] = {
                "order": [p, d, q],
                "aic": model_fit.aic,
                "bic": model_fit.bic
            }
            
            # Store performance metrics
            result["performance_metrics"] = {
                "aic": model_fit.aic,
                "bic": model_fit.bic
            }
            
        except Exception as e:
            logger.error(f"Error in ARIMA forecasting: {e}")
            result["error"] = f"ARIMA forecasting failed: {e}"
            # Fallback to SMA forecasting
            self._generate_sma_forecast(ts_data, periods, result)
            
    def _auto_arima_order(self, ts_data: pd.Series) -> Tuple[int, int, int]:
        """
        Simple automatic order selection for ARIMA.
        
        Args:
            ts_data: Time series data
            
        Returns:
            Tuple of (p, d, q) order parameters
        """
        # Check for stationarity and determine d
        result = adfuller(ts_data.dropna())
        
        # If p-value > 0.05, the series is non-stationary
        d = 0
        if result[1] > 0.05:
            d = 1
            # Try first difference
            diff1 = ts_data.diff().dropna()
            result = adfuller(diff1)
            if result[1] > 0.05:
                d = 2  # If still non-stationary, use second difference
                
        # Simple heuristic for p and q
        p = min(5, int(len(ts_data) / 10) or 1)
        q = min(2, int(len(ts_data) / 20) or 1)
        
        return p, d, q
        
    def _generate_sma_forecast(self, ts_data: pd.Series, periods: int, result: Dict[str, Any]) -> None:
        """
        Generate forecast using Simple Moving Average.
        
        Args:
            ts_data: Time series data
            periods: Number of periods to forecast
            result: Dictionary to store results (modified in-place)
        """
        try:
            # Use the mean of the last 30% of the data or at least 5 points
            window = max(5, int(len(ts_data) * 0.3))
            last_values = ts_data.tail(window)
            forecast_value = last_values.mean()
            
            # Generate forecast dates
            forecast_dates = pd.date_range(start=ts_data.index[-1], periods=periods+1, freq=self._infer_frequency(ts_data))[1:]
            
            # Store forecast as constant value
            forecast_values = [forecast_value] * periods
            
            # Calculate simple confidence interval based on historical standard deviation
            std_dev = last_values.std()
            z_value = 1.96  # ~95% confidence interval
            lower_ci = [forecast_value - z_value * std_dev] * periods
            upper_ci = [forecast_value + z_value * std_dev] * periods
            
            result["forecast"] = {
                "dates": [str(d) for d in forecast_dates],
                "values": forecast_values
            }
            
            result["confidence_intervals"] = {
                "lower": lower_ci,
                "upper": upper_ci
            }
            
            result["model_params"] = {
                "window_size": window
            }
            
            result["model"] = "sma"  # Overwrite model in result
            
        except Exception as e:
            logger.error(f"Error in SMA forecasting: {e}")
            result["error"] = f"SMA forecasting failed: {e}"
            
    def _infer_frequency(self, ts_data: pd.Series) -> str:
        """
        Infer frequency of time series data.
        
        Args:
            ts_data: Time series data
            
        Returns:
            Frequency string (e.g., 'D', 'M', etc.)
        """
        # Try pandas frequency inference
        freq = pd.infer_freq(ts_data.index)
        
        if freq is None:
            # Simple heuristic based on average difference
            if len(ts_data) < 2:
                return 'D'  # Default to daily if not enough data
                
            # Calculate average time delta in seconds
            time_deltas = np.diff(ts_data.index.astype(np.int64)) / 1e9
            avg_delta = np.mean(time_deltas)
            
            # Map to common frequencies
            if avg_delta < 60:
                freq = 'S'  # Second
            elif avg_delta < 3600:
                freq = 'T'  # Minute
            elif avg_delta < 86400:
                freq = 'H'  # Hour
            elif avg_delta < 86400 * 7:
                freq = 'D'  # Day
            elif avg_delta < 86400 * 31:
                freq = 'W'  # Week
            else:
                freq = 'M'  # Month
                
        return freq or 'D'  # Default to daily if inference fails
            
    def _generate_forecast_chart(self, historical_data: pd.Series, 
                               result: Dict[str, Any],
                               target_column: str) -> str:
        """
        Generate a chart visualizing the forecast results.
        
        Args:
            historical_data: Historical time series data
            result: Dictionary containing forecast results
            target_column: Name of the target column
            
        Returns:
            Path to the saved chart
        """
        try:
            # Create figure
            plt.figure(figsize=(12, 6))
            
            # Plot historical data
            plt.plot(historical_data.index, historical_data.values, label='Historical Data', color='blue')
            
            # Extract forecast data
            forecast_dates = [pd.to_datetime(d) for d in result["forecast"]["dates"]]
            forecast_values = result["forecast"]["values"]
            
            # Plot forecast
            plt.plot(forecast_dates, forecast_values, label='Forecast', color='red')
            
            # Plot confidence intervals if available
            if "confidence_intervals" in result:
                lower_bound = result["confidence_intervals"]["lower"]
                upper_bound = result["confidence_intervals"]["upper"]
                plt.fill_between(forecast_dates, lower_bound, upper_bound, color='red', alpha=0.2)
                
            # Set labels and title
            plt.title(f"Time Series Forecast for {target_column}")
            plt.xlabel("Date")
            plt.ylabel(target_column)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Format x-axis dates
            plt.gcf().autofmt_xdate()
            
            # Save chart
            os.makedirs(os.path.join(self.output_dir, "charts"), exist_ok=True)
            chart_filename = f"{target_column.lower().replace(' ', '_')}_forecast.png"
            chart_path = os.path.join(self.output_dir, "charts", chart_filename)
            plt.savefig(chart_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"Error generating forecast chart: {e}")
            return "chart_generation_failed" 