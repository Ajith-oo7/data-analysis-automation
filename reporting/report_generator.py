import os
import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import datetime
from jinja2 import Environment, FileSystemLoader, select_autoescape
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np  # Add numpy import

# Import email sender
from .email_sender import EmailSender

# Mock imports for demonstration purposes
# In a real implementation, these would be actual imports
class MockTableauClient:
    def __init__(self, server, username, password, site):
        self.server = server
        self.username = username
        self.password = password
        self.site = site
    
    def publish_report(self, data, report_name):
        return f"https://{self.server}/views/{report_name}"

class MockPowerBIClient:
    def __init__(self, workspace_id, report_id):
        self.workspace_id = workspace_id
        self.report_id = report_id
    
    def publish_report(self, data, report_name):
        return f"https://app.powerbi.com/groups/{self.workspace_id}/reports/{report_name}"

# Load environment variables for credentials
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', '.env'))

# Configure logging
logger = logging.getLogger(__name__)

def generate_html_report(data: pd.DataFrame, insights: Dict[str, Any], use_cases: List[Dict[str, Any]], config: Dict[str, Any]) -> str:
    """
    Generate an HTML report.
    
    Args:
        data: Input DataFrame
        insights: Dictionary of insights
        use_cases: List of use cases
        config: Configuration parameters
        
    Returns:
        Path to the generated HTML report
    """
    try:
        logger.info("Generating HTML report")
        
        # Create report directory if it doesn't exist
        report_dir = config.get("output_dir", "reports")
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate timestamp for report name
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"data_analysis_report_{timestamp}.html"
        report_path = os.path.join(report_dir, report_name)
        
        # Create some basic data visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Data summary stats
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Data Analysis Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #3498db;
            margin-top: 30px;
        }}
        .insight-box {{
            background-color: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin-bottom: 20px;
        }}
        .use-case {{
            background-color: #f0f7fb;
            border: 1px solid #d0e3ef;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
        }}
        .use-case h3 {{
            margin-top: 0;
            color: #2980b9;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 20px;
        }}
        th, td {{
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        .high {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .medium {{
            color: #f39c12;
            font-weight: bold;
        }}
        .low {{
            color: #27ae60;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Data Analysis Report</h1>
        <p>Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <h2>Data Summary</h2>
        <table>
            <tr>
                <th>Total Records</th>
                <td>{len(data)}</td>
            </tr>
            <tr>
                <th>Total Columns</th>
                <td>{len(data.columns)}</td>
            </tr>
        </table>
        
        <h2>Key Insights</h2>
"""
        
        # Add insights
        for insight_type, insight_text in insights.items():
            html_content += f"""
        <div class="insight-box">
            <h3>{insight_type.capitalize()} Insights</h3>
            <pre>{insight_text}</pre>
        </div>
"""
        
        # Add use cases
        html_content += """
        <h2>Recommended Use Cases</h2>
"""
        for use_case in use_cases:
            impact_class = use_case.get("impact", "medium").lower()
            html_content += f"""
        <div class="use-case">
            <h3>{use_case.get("title", "Unnamed Use Case")}</h3>
            <p>{use_case.get("description", "No description provided.")}</p>
            <p>Impact: <span class="{impact_class}">{use_case.get("impact", "Medium")}</span> | Effort: {use_case.get("effort", "Medium")}</p>
        </div>
"""

        # Close HTML
        html_content += """
    </div>
</body>
</html>
"""
        
        # Write HTML to file
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {report_path}")
        return report_path
    
    except Exception as e:
        logger.error(f"Error generating HTML report: {str(e)}")
        raise

def generate_pdf_report(html_path: str, config: Dict[str, Any]) -> Optional[str]:
    """
    Generate a PDF report from HTML.
    
    Args:
        html_path: Path to the HTML report
        config: Configuration parameters
        
    Returns:
        Path to the PDF report or None if generation fails
    """
    try:
        # Check if we have the required libraries
        try:
            import weasyprint
        except ImportError:
            logger.warning("weasyprint library not installed. Please install with: pip install weasyprint")
            return None
        
        pdf_path = html_path.replace('.html', '.pdf')
        
        # Convert HTML to PDF
        weasyprint.HTML(filename=html_path).write_pdf(pdf_path)
        
        logger.info(f"PDF report generated: {pdf_path}")
        return pdf_path
        
    except Exception as e:
        logger.error(f"Error generating PDF report: {str(e)}")
        return None

def generate_report(data: pd.DataFrame, insights_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a report from data and insights.
    
    Args:
        data: Input DataFrame
        insights_path: Path to the insights JSON file
        config: Configuration parameters
        
    Returns:
        Dictionary with report generation results
    """
    result = {
        "status": "success",
        "message": "Report generated successfully",
        "report_paths": [],
    }
    
    try:
        # Load insights
        with open(insights_path, 'r') as f:
            insights_data = json.load(f)
        
        insights = insights_data.get("insights", {})
        use_cases = insights_data.get("use_cases", [])
        
        # Generate HTML report
        report_dir = config.get("reporting", {}).get("output_dir", "reports")
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate filename from insights path
        base_name = os.path.basename(insights_path)
        file_name = os.path.splitext(base_name)[0].replace("_insights", "_report")
        
        # Generate HTML report
        html_content = generate_html_report(data, insights, use_cases, config)
        html_path = os.path.join(report_dir, f"{file_name}.html")
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        result["report_paths"].append(html_path)
        logger.info(f"HTML report generated: {html_path}")
        
        # Generate PDF report if enabled
        if config.get("reporting", {}).get("generate_pdf", False):
            pdf_path = generate_pdf_report(html_path, config)
            if pdf_path:
                result["report_paths"].append(pdf_path)
                logger.info(f"PDF report generated: {pdf_path}")
        
        # Send email notification if enabled
        if config.get("reporting", {}).get("email", {}).get("enabled", False):
            email_config = config.get("reporting", {}).get("email", {})
            recipients = email_config.get("recipients", [])
            
            if recipients:
                # Extract data summary for email
                data_summary = {
                    "file_name": file_name,
                    "num_rows": len(data),
                    "num_columns": len(data.columns),
                    "missing_percentage": round(data.isna().sum().sum() / (data.shape[0] * data.shape[1]) * 100, 2)
                }
                
                # Create email sender and send notification
                email_sender = EmailSender(config)
                email_result = email_sender.send_report_notification(
                    recipients=recipients,
                    subject=f"Data Analysis Report: {file_name}",
                    message="A new data analysis report has been generated and is ready for your review.",
                    report_paths=result["report_paths"],
                    data_summary=data_summary
                )
                
                if email_result["status"] == "success":
                    logger.info(f"Email notification sent to {len(recipients)} recipients")
                else:
                    logger.warning(f"Failed to send email notification: {email_result['message']}")
                
                result["email_notification"] = email_result["status"]
        
    except Exception as e:
        result["status"] = "error"
        result["message"] = f"Error generating report: {str(e)}"
        logger.error(f"Error generating report: {str(e)}")
    
    return result

class ReportGenerator:
    """
    Main class for generating reports from data and insights.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the report generator with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.report_config = config.get("reporting", {})
        self.output_dir = self.report_config.get("output_dir", "reports")
        self.charts_dir = os.path.join(self.output_dir, "charts")
        self.template_dir = os.path.join(os.path.dirname(__file__), "templates")
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.charts_dir, exist_ok=True)
        
        # Initialize Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(self.template_dir),
            autoescape=select_autoescape(['html', 'xml'])
        )
        
        # Set up chart parameters
        self.chart_theme = self.report_config.get("chart_theme", "default")
        self.color_palette = self.report_config.get("color_palette", ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"])
        self.include_charts = self.report_config.get("include_charts", True)
        
        # Initialize additional charts dictionary
        self.additional_charts = {}
        
        # Set chart style
        if self.chart_theme == "dark":
            plt.style.use('dark_background')
        else:
            plt.style.use('seaborn-pastel')
            
        sns.set_palette(self.color_palette)
    
    def add_chart(self, chart_name: str, chart_path: str) -> None:
        """
        Add an external chart to include in the report.
        
        Args:
            chart_name: Name identifier for the chart
            chart_path: Path to the chart image file
        """
        # Check if the chart file exists
        if os.path.exists(chart_path):
            self.additional_charts[chart_name] = chart_path
            logger.info(f"Added external chart '{chart_name}' to report: {chart_path}")
        else:
            logger.warning(f"Chart file not found: {chart_path}")
    
    def generate_report(self, data: pd.DataFrame, insights_path: str, file_path: str) -> Dict[str, Any]:
        """
        Generate a report from data and insights.
        
        Args:
            data: Input DataFrame
            insights_path: Path to the insights JSON file
            file_path: Original data file path
            
        Returns:
            Dictionary with report generation results
        """
        result = {
            "status": "success",
            "message": "Report generated successfully",
            "report_paths": []
        }
        
        try:
            # Load insights
            with open(insights_path, 'r') as f:
                insights_data = json.load(f)
            
            insights = insights_data.get("insights", {})
            use_cases = insights_data.get("use_cases", [])
            executive_summary = insights_data.get("executive_summary", "")
            
            # Generate charts
            chart_paths = {}
            if self.include_charts:
                chart_paths = self._generate_charts(data)
                
            # Add any forecasting charts
            if "forecast" in self.additional_charts:
                chart_paths["forecast"] = self.additional_charts["forecast"]
                
            # Generate HTML report
            base_name = os.path.basename(file_path)
            file_name = os.path.splitext(base_name)[0]
            
            html_path = os.path.join(self.output_dir, f"{file_name}_report.html")
            
            # Prepare template context
            context = {
                "title": f"Data Analysis Report - {file_name}",
                "generation_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "data_summary": self._generate_data_summary(data, file_path),
                "insights": insights,
                "use_cases": use_cases,
                "chart_paths": chart_paths,
                "executive_summary": executive_summary,
                "has_forecast": "forecast" in chart_paths
            }
            
            # Get template
            template = self._get_report_template()
            
            # Render template
            html_content = template.render(**context)
            
            # Write HTML to file
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            result["report_paths"].append(html_path)
            logger.info(f"HTML report generated: {html_path}")
            
            # Generate PDF report if enabled
            if self.report_config.get("generate_pdf", False):
                pdf_path = self._generate_pdf_report(html_path)
                if pdf_path:
                    result["report_paths"].append(pdf_path)
                    logger.info(f"PDF report generated: {pdf_path}")
            
            # Send email notification if enabled
            if self.report_config.get("email", {}).get("enabled", False) and self.report_config.get("email", {}).get("send_on_completion", True):
                self._send_email_notification(result["report_paths"], context["data_summary"])
                
        except Exception as e:
            result["status"] = "error"
            result["message"] = f"Error generating report: {str(e)}"
            logger.error(f"Error generating report: {str(e)}")
        
        return result
    
    def _generate_charts(self, data: pd.DataFrame) -> Dict[str, str]:
        """
        Generate charts from data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Dictionary of chart paths
        """
        chart_paths = {}
        
        try:
            # Skip chart generation if no numerical columns
            numeric_columns = data.select_dtypes(include=['number']).columns
            if len(numeric_columns) == 0:
                logger.warning("No numeric columns found for chart generation")
                return chart_paths
            
            # Create histogram for numeric columns
            plt.figure(figsize=(12, 8))
            
            # Select up to 5 numeric columns for visualization
            columns_to_plot = numeric_columns[:5]
            
            for i, column in enumerate(columns_to_plot):
                plt.subplot(2, 3, i+1)
                sns.histplot(data[column].dropna(), kde=True, color=self.color_palette[i % len(self.color_palette)])
                plt.title(f'Distribution of {column}')
                plt.tight_layout()
            
            # Save chart
            hist_path = os.path.join(self.charts_dir, "distribution.png")
            plt.savefig(hist_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            chart_paths["distribution"] = hist_path
            
            # Create correlation heatmap if multiple numeric columns
            if len(numeric_columns) > 1:
                plt.figure(figsize=(10, 8))
                correlation = data[numeric_columns].corr()
                sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
                plt.title('Correlation Matrix')
                
                # Save chart
                corr_path = os.path.join(self.charts_dir, "correlation.png")
                plt.savefig(corr_path, dpi=100, bbox_inches='tight')
                plt.close()
                
                chart_paths["correlation"] = corr_path
            
            # Create boxplot for numeric columns
            plt.figure(figsize=(12, 8))
            
            for i, column in enumerate(columns_to_plot):
                plt.subplot(2, 3, i+1)
                sns.boxplot(y=data[column], color=self.color_palette[i % len(self.color_palette)])
                plt.title(f'Boxplot of {column}')
                plt.tight_layout()
            
            # Save chart
            box_path = os.path.join(self.charts_dir, "boxplot.png")
            plt.savefig(box_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            chart_paths["boxplot"] = box_path
            
            # Create pairplot if fewer than 5 numeric columns
            if 2 <= len(numeric_columns) <= 4:
                plt.figure(figsize=(12, 10))
                pair_plot = sns.pairplot(data[numeric_columns])
                
                # Save chart
                pair_path = os.path.join(self.charts_dir, "pairplot.png")
                pair_plot.savefig(pair_path, dpi=100)
                plt.close()
                
                chart_paths["pairplot"] = pair_path
            
            # Create bar chart for categorical columns if any
            categorical_columns = data.select_dtypes(include=['object', 'category']).columns
            if len(categorical_columns) > 0:
                for i, column in enumerate(categorical_columns[:2]):  # Limit to 2 categorical columns
                    plt.figure(figsize=(10, 6))
                    value_counts = data[column].value_counts().nlargest(10)  # Top 10 values
                    sns.barplot(x=value_counts.index, y=value_counts.values, palette=self.color_palette)
                    plt.title(f'Top values for {column}')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    # Save chart
                    cat_path = os.path.join(self.charts_dir, f"category_{i}.png")
                    plt.savefig(cat_path, dpi=100, bbox_inches='tight')
                    plt.close()
                    
                    chart_paths[f"category_{i}"] = cat_path
                    
            logger.info(f"Generated {len(chart_paths)} charts for report")
            return chart_paths
            
        except Exception as e:
            logger.error(f"Error generating charts: {str(e)}")
            return chart_paths
    
    def _generate_data_summary(self, data: pd.DataFrame, file_path: str) -> Dict[str, Any]:
        """
        Generate a summary of the data.
        
        Args:
            data: Input DataFrame
            file_path: Original data file path
            
        Returns:
            Dictionary with data summary
        """
        summary = {
            "file_name": os.path.basename(file_path),
            "file_path": file_path,
            "file_size_kb": round(os.path.getsize(file_path) / 1024, 2),
            "rows": len(data),
            "columns": len(data.columns),
            "column_list": list(data.columns),
            "missing_values": int(data.isna().sum().sum()),
            "missing_percentage": round(data.isna().sum().sum() / (data.shape[0] * data.shape[1]) * 100, 2),
            "numeric_columns": list(data.select_dtypes(include=['number']).columns),
            "categorical_columns": list(data.select_dtypes(include=['object', 'category']).columns),
            "date_columns": list(data.select_dtypes(include=['datetime']).columns)
        }
        
        # Add descriptive statistics for numeric columns
        if summary["numeric_columns"]:
            stats = data[summary["numeric_columns"]].describe().to_dict()
            summary["stats"] = stats
        
        return summary
    
    def _get_report_template(self) -> Any:
        """
        Get the report template.
        
        Returns:
            Jinja2 template
        """
        try:
            # Try to load template from templates directory
            if os.path.exists(os.path.join(self.template_dir, "report_template.html")):
                return self.env.get_template("report_template.html")
            else:
                # Fall back to default template
                logger.warning("Template file not found, using default template")
                return self._create_default_template()
        except Exception as e:
            logger.error(f"Error loading template: {str(e)}")
            return self._create_default_template()
    
    def _create_default_template(self) -> Any:
        """
        Create a default report template.
        
        Returns:
            Jinja2 template
        """
        template_str = """<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 10px;
        }
        h2 {
            color: #3498db;
            margin-top: 30px;
        }
        .insight-box {
            background-color: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin-bottom: 20px;
        }
        .use-case {
            background-color: #f0f7fb;
            border: 1px solid #d0e3ef;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
        }
        .use-case h3 {
            margin-top: 0;
            color: #2980b9;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 20px;
        }
        th, td {
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f2f2f2;
        }
        .chart-container {
            margin: 30px 0;
            text-align: center;
        }
        .chart-container img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .summary-box {
            background-color: #f9f9f9;
            border: 1px solid #ddd;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 5px;
        }
        .forecast-section {
            background-color: #f5f5f5;
            border: 1px solid #ddd;
            padding: 20px;
            margin: 30px 0;
            border-radius: 5px;
        }
        .forecast-section h2 {
            color: #e74c3c;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>{{ title }}</h1>
        <p>Generated on {{ generation_time }}</p>
        
        {% if executive_summary %}
        <div class="summary-box">
            <h2>Executive Summary</h2>
            <p>{{ executive_summary }}</p>
        </div>
        {% endif %}
        
        <h2>Data Summary</h2>
        <table>
            <tr>
                <th>File Name</th>
                <td>{{ data_summary.file_name }}</td>
            </tr>
            <tr>
                <th>File Size</th>
                <td>{{ data_summary.file_size_kb }} KB</td>
            </tr>
            <tr>
                <th>Total Records</th>
                <td>{{ data_summary.rows }}</td>
            </tr>
            <tr>
                <th>Total Columns</th>
                <td>{{ data_summary.columns }}</td>
            </tr>
            <tr>
                <th>Missing Values</th>
                <td>{{ data_summary.missing_values }} ({{ data_summary.missing_percentage }}%)</td>
            </tr>
        </table>
        
        {% if chart_paths %}
        <h2>Data Visualizations</h2>
        
        {% if 'distribution' in chart_paths %}
        <div class="chart-container">
            <h3>Distribution of Numeric Variables</h3>
            <img src="{{ chart_paths.distribution }}" alt="Distribution Chart">
        </div>
        {% endif %}
        
        {% if 'correlation' in chart_paths %}
        <div class="chart-container">
            <h3>Correlation Matrix</h3>
            <img src="{{ chart_paths.correlation }}" alt="Correlation Matrix">
        </div>
        {% endif %}
        
        {% if 'boxplot' in chart_paths %}
        <div class="chart-container">
            <h3>Boxplots of Numeric Variables</h3>
            <img src="{{ chart_paths.boxplot }}" alt="Boxplot Chart">
        </div>
        {% endif %}
        
        {% if 'pairplot' in chart_paths %}
        <div class="chart-container">
            <h3>Pair Plot</h3>
            <img src="{{ chart_paths.pairplot }}" alt="Pair Plot">
        </div>
        {% endif %}
        
        {% for i in range(5) %}
            {% set cat_key = 'category_' ~ i %}
            {% if cat_key in chart_paths %}
            <div class="chart-container">
                <h3>Categorical Variable Distribution</h3>
                <img src="{{ chart_paths[cat_key] }}" alt="Category Chart">
            </div>
            {% endif %}
        {% endfor %}
        {% endif %}
        
        {% if has_forecast %}
        <div class="forecast-section">
            <h2>Time Series Forecast</h2>
            <p>The following chart shows the forecast of future values based on the historical data patterns.</p>
            <div class="chart-container">
                <img src="{{ chart_paths.forecast }}" alt="Forecast Chart">
            </div>
            <p>Note: The forecast is based on the patterns observed in the historical data. The confidence intervals represent the range of possible future values.</p>
        </div>
        {% endif %}
        
        <h2>Key Insights</h2>
        {% for insight_type, insight_text in insights.items() %}
        <div class="insight-box">
            <h3>{{ insight_type|capitalize }} Insights</h3>
            <pre>{{ insight_text }}</pre>
        </div>
        {% endfor %}
        
        <h2>Recommended Use Cases</h2>
        {% for use_case in use_cases %}
        <div class="use-case">
            <h3>{{ use_case.title|default('Unnamed Use Case') }}</h3>
            <p>{{ use_case.description|default('No description provided.') }}</p>
            <p>Impact: <span class="{{ use_case.impact|default('medium')|lower }}">{{ use_case.impact|default('Medium') }}</span> | Effort: {{ use_case.effort|default('Medium') }}</p>
        </div>
        {% endfor %}
    </div>
</body>
</html>
"""
        return self.env.from_string(template_str)
    
    def _generate_pdf_report(self, html_path: str) -> Optional[str]:
        """
        Generate a PDF report from HTML.
        
        Args:
            html_path: Path to the HTML report
            
        Returns:
            Path to the PDF report or None if generation fails
        """
        try:
            # Check if we have the required libraries
            try:
                import weasyprint
            except ImportError:
                logger.warning("weasyprint library not installed. Please install with: pip install weasyprint")
                return None
            
            pdf_path = html_path.replace('.html', '.pdf')
            
            # Convert HTML to PDF
            weasyprint.HTML(filename=html_path).write_pdf(pdf_path)
            
            logger.info(f"PDF report generated: {pdf_path}")
            return pdf_path
            
        except ImportError:
            logger.warning("weasyprint library not installed. Please install with: pip install weasyprint")
            return None
        except Exception as e:
            logger.error(f"Error generating PDF report: {str(e)}")
            return None
    
    def _send_email_notification(self, report_paths: List[str], data_summary: Dict[str, Any]) -> bool:
        """
        Send email notification with report.
        
        Args:
            report_paths: List of paths to the generated reports
            data_summary: Dictionary with data summary
            
        Returns:
            True if email sent successfully, False otherwise
        """
        try:
            email_config = self.report_config.get("email", {})
            recipients = email_config.get("recipients", [])
            
            if not recipients:
                logger.warning("No email recipients specified. Email notification skipped.")
                return False
            
            # Create email sender and send notification
            email_sender = EmailSender(self.config)
            file_name = data_summary.get("file_name", "Unknown")
            
            email_result = email_sender.send_report_notification(
                recipients=recipients,
                subject=f"Data Analysis Report: {file_name}",
                message="A new data analysis report has been generated and is ready for your review.",
                report_paths=report_paths,
                data_summary=data_summary
            )
            
            if email_result["status"] == "success":
                logger.info(f"Email notification sent to {len(recipients)} recipients")
                return True
            else:
                logger.warning(f"Failed to send email notification: {email_result['message']}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending email notification: {str(e)}")
            return False

# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Import locally to avoid circular imports
    import numpy as np
    
    # Sample configuration
    sample_config = {
        "reporting": {
            "output_dir": "reports",
            "generate_pdf": False
        }
    }
    
    # Create a sample DataFrame
    data = {
        'id': range(1, 11),
        'name': ['John Smith', 'Jane Doe', 'Mike Johnson', 'Sarah Williams', 'Robert Brown', 
                 None, 'Emily Davis', 'David Wilson', 'Lisa Anderson', 'Mark Taylor'],
        'age': [35, 28, 42, 31, 39, 45, 29, 33, 37, 41],
        'salary': [75000, 82000, 95000, None, 110000, 120000, 78000, 88000, None, 115000],
        'department': ['Engineering', 'Marketing', 'Engineering', 'HR', 'Finance', 
                      'Management', 'Marketing', 'Engineering', 'Finance', 'Management'],
        'hire_date': ['2018-05-12', '2019-03-24', '2015-11-01', '2020-01-15', '2017-08-30',
                     '2016-04-22', '2021-06-10', '2019-09-05', '2018-12-18', '2017-02-28']
    }
    sample_df = pd.DataFrame(data)
    
    # Create sample insights file
    sample_insights = {
        "file_name": "sample_data.csv",
        "insights": {
            "general": "General insights about the data...",
            "salary": "Salary distribution insights...",
            "department": "Department distribution insights..."
        },
        "use_cases": [
            {
                "title": "Salary Review Process",
                "description": "Implement a data-driven salary review process.",
                "impact": "High",
                "effort": "Medium"
            },
            {
                "title": "Department Expansion Planning",
                "description": "Use data insights for department expansion planning.",
                "impact": "Medium",
                "effort": "Low"
            }
        ]
    }
    
    # Create insights directory and save sample insights
    os.makedirs("insights", exist_ok=True)
    insights_path = "insights/sample_data_insights.json"
    with open(insights_path, 'w') as f:
        json.dump(sample_insights, f, indent=2)
    
    # Generate report
    generator = ReportGenerator(sample_config)
    result = generator.generate_report(sample_df, insights_path, "sample_data.csv")
    
    if result["status"] == "success":
        print("Report generated successfully!")
        print("\nReport paths:")
        for path in result["report_paths"]:
            print(f"- {path}")
    else:
        print(f"Report generation failed: {result['message']}") 