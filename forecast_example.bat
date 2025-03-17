@echo off
echo Running forecasting example with sample time series data...
echo.

python main.py --data-source source_data/sample_timeseries.csv --mode full --forecast --date-column date --target-column sales --forecast-periods 10 --verbose

echo.
echo Example complete!
pause 