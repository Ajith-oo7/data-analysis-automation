@echo off
echo ===== Snowflake Connection Tester =====
echo.

echo Installing required packages...
pip install python-dotenv snowflake-connector-python pandas

echo.
echo Running Snowflake connection test...
python test_snowflake_connection.py

echo.
echo Test completed. Press any key to exit.
pause > nul 