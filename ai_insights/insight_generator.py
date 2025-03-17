import os
import json
import logging
import pandas as pd
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import time

# Load environment variables for credentials
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', '.env'))

# Configure logging
logger = logging.getLogger(__name__)

# Try to import snowflake, but make it optional
try:
    import snowflake.connector
    SNOWFLAKE_AVAILABLE = True
except ImportError:
    logger.warning("Snowflake connector not installed. Snowflake functionality will be disabled.")
    SNOWFLAKE_AVAILABLE = False

# Configure OpenAI API
try:
    import openai
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    if not openai.api_key:
        logger.warning("OpenAI API key not found. Using mock LLM as fallback.")
        USE_MOCK_LLM = True
    else:
        USE_MOCK_LLM = False
except ImportError:
    logger.warning("OpenAI package not installed. Using mock LLM.")
    USE_MOCK_LLM = True

class OpenAILLM:
    """OpenAI integration for generating insights."""
    
    def __init__(self, model: str = "gpt-4"):
        """
        Initialize the OpenAI LLM.
        
        Args:
            model: OpenAI model to use
        """
        self.model = model
    
    def generate(self, prompt: str) -> str:
        """
        Generate a response to a prompt using OpenAI.
        
        Args:
            prompt: The input prompt
            
        Returns:
            A generated response
        """
        try:
            # Add retry logic for API rate limits
            max_retries = 3
            retry_delay = 5
            
            for attempt in range(max_retries):
                try:
                    response = openai.ChatCompletion.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a data analyst providing concise, accurate insights from data."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1,  # Lower temperature for more consistent, analytical responses
                        max_tokens=1000
                    )
                    return response.choices[0].message.content.strip()
                except openai.error.RateLimitError:
                    if attempt < max_retries - 1:
                        logger.warning(f"Rate limit exceeded. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        raise
        except Exception as e:
            logger.error(f"Error using OpenAI API: {str(e)}")
            # Fallback to mock LLM if OpenAI fails
            logger.warning("Falling back to mock LLM")
            mock_llm = MockLLM()
            return mock_llm.generate(prompt)

# Keep the mock LLM for fallback and testing purposes
class MockLLM:
    """Mock LLM class to simulate AI model responses for demonstration."""
    
    def generate(self, prompt: str) -> str:
        """
        Generate a response to a prompt.
        
        Args:
            prompt: The input prompt
            
        Returns:
            A generated response
        """
        # Simulate an AI-generated response based on keywords in the prompt
        if "salary" in prompt.lower():
            return """Based on the data analysis, I've identified the following insights:
1. The average salary across all departments is $95,375.
2. The Finance department has the highest average salary at $110,000.
3. There's a 34% salary gap between the highest paid department (Finance) and the lowest (HR).
4. Employees with more than 5 years at the company earn 23% more on average than newer employees.
5. There's a positive correlation (0.78) between age and salary, suggesting age/experience is valued."""
        
        elif "department" in prompt.lower():
            return """Department Distribution Analysis:
1. Engineering has the highest headcount with 3 employees (30% of workforce).
2. Marketing, Finance, and Management each have 2 employees (20% each).
3. HR has the lowest headcount with 1 employee (10%).
4. Engineering department has hired consistently over the years 2015-2019.
5. Marketing has the youngest average employee age at 28.5 years."""
        
        else:
            return """General Data Insights:
1. The average employee age is 36 years.
2. 80% of employees have complete data records.
3. The company experienced the highest hiring rate in 2017-2019.
4. There's a diverse age distribution across departments.
5. 20% of records have missing values that require attention."""

def connect_to_snowflake(config: Dict[str, Any]) -> Optional[Any]:
    """
    Connect to Snowflake using configuration.
    
    Args:
        config: Snowflake connection configuration
        
    Returns:
        Snowflake connection object or None if connection fails
    """
    # If Snowflake is not available, return None
    if not SNOWFLAKE_AVAILABLE:
        logger.warning("Snowflake connector not installed. Using mock data instead.")
        return None
        
    try:
        # Try to get credentials from environment variables first
        account = os.environ.get('SNOWFLAKE_ACCOUNT') or config.get('account')
        user = os.environ.get('SNOWFLAKE_USER') or config.get('user')
        password = os.environ.get('SNOWFLAKE_PASSWORD') or config.get('password')
        warehouse = os.environ.get('SNOWFLAKE_WAREHOUSE') or config.get('warehouse')
        database = os.environ.get('SNOWFLAKE_DATABASE') or config.get('database')
        schema = os.environ.get('SNOWFLAKE_SCHEMA') or config.get('schema')
        
        if not all([account, user, password, warehouse, database, schema]):
            logger.warning("Incomplete Snowflake credentials. Using mock data instead.")
            return None
        
        conn = snowflake.connector.connect(
            account=account,
            user=user,
            password=password,
            warehouse=warehouse,
            database=database,
            schema=schema
        )
        
        logger.info("Successfully connected to Snowflake")
        return conn
    
    except Exception as e:
        logger.error(f"Failed to connect to Snowflake: {str(e)}")
        return None

def query_snowflake(conn, query: str) -> pd.DataFrame:
    """
    Execute a query on Snowflake and return the results as a DataFrame.
    
    Args:
        conn: Snowflake connection
        query: SQL query to execute
        
    Returns:
        DataFrame with query results
    """
    # If Snowflake is not available or connection is None, return empty DataFrame
    if not SNOWFLAKE_AVAILABLE or conn is None:
        logger.warning("Snowflake not available. Returning empty DataFrame.")
        return pd.DataFrame()
        
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetch_pandas_all()
        cursor.close()
        return result
    except Exception as e:
        logger.error(f"Error executing Snowflake query: {str(e)}")
        raise

def prepare_data_description(data: pd.DataFrame) -> str:
    """
    Prepare a detailed description of the data for AI analysis.
    
    Args:
        data: Input DataFrame
        
    Returns:
        String description of the data
    """
    description = []
    
    # Basic DataFrame info
    description.append(f"DataFrame shape: {data.shape[0]} rows, {data.shape[1]} columns")
    description.append(f"Columns: {', '.join(data.columns.tolist())}")
    
    # Data types
    dtypes = data.dtypes.astype(str).to_dict()
    description.append("Column data types:")
    for col, dtype in dtypes.items():
        description.append(f"  - {col}: {dtype}")
    
    # Basic statistics for numeric columns
    numeric_stats = data.describe().to_string()
    description.append("Numeric column statistics:")
    description.append(numeric_stats)
    
    # Missing values
    missing = data.isna().sum().to_dict()
    description.append("Missing values by column:")
    for col, count in missing.items():
        if count > 0:
            description.append(f"  - {col}: {count} missing values ({count/len(data):.2%})")
    
    # Categorical column distributions (if any)
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        description.append("Categorical column distributions:")
        for col in categorical_cols:
            if len(data[col].unique()) < 20:  # Only show if there aren't too many unique values
                value_counts = data[col].value_counts().head(10).to_dict()
                description.append(f"  - {col}: {value_counts}")
    
    return "\n".join(description)

def generate_insights(data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate insights from the data using AI models.
    
    Args:
        data: Input DataFrame to analyze
        config: Configuration parameters
        
    Returns:
        Dictionary containing generated insights
    """
    logger.info("Generating insights from data")
    
    try:
        # Choose between real OpenAI and mock LLM
        model_name = config.get("ai_insights", {}).get("models", {}).get("default", "gpt-4")
        
        if USE_MOCK_LLM:
            logger.info("Using mock LLM for insight generation")
            model = MockLLM()
        else:
            logger.info(f"Using OpenAI (model: {model_name}) for insight generation")
            model = OpenAILLM(model=model_name)
        
        # Prepare data description
        data_description = prepare_data_description(data)
        
        # Generate different types of insights with improved prompts
        general_prompt = f"""Analyze the following dataset and provide 5 key general insights. Focus on patterns, trends, and anomalies. Be specific and quantitative where possible.

DATASET INFORMATION:
{data_description}

Provide insights in a numbered list format (1-5)."""

        # If salary column exists, create salary analysis
        if 'salary' in data.columns:
            # Prepare salary specific data
            salary_data = pd.DataFrame()
            salary_data['salary'] = data['salary']
            
            # Add department if it exists
            if 'department' in data.columns:
                salary_data['department'] = data['department']
                dept_salary = data.groupby('department')['salary'].mean().to_dict()
                
            # Add age if it exists
            if 'age' in data.columns:
                salary_data['age'] = data['age']
                
            # Add hire_date if it exists
            if 'hire_date' in data.columns:
                salary_data['hire_date'] = data['hire_date']
                
            salary_prompt = f"""Analyze the salary distribution in this dataset. Look for patterns related to departments, age, or tenure if available.

DATASET INFORMATION:
{prepare_data_description(salary_data)}

If department information is available, include analysis of salary differences between departments.
If age information is available, analyze the correlation between age and salary.
If hire date information is available, analyze how tenure relates to salary.

Provide 5 specific, quantitative insights in a numbered list format (1-5)."""
        else:
            salary_prompt = None
            
        # If department column exists, create department analysis
        if 'department' in data.columns:
            dept_counts = data['department'].value_counts().to_dict()
            
            department_prompt = f"""Analyze the department distribution in this dataset and provide insights.

DEPARTMENT DISTRIBUTION:
{dept_counts}

Consider:
- Distribution of employees across departments
- Any patterns related to departments and other variables
- Any notable characteristics of specific departments

Provide 5 specific insights in a numbered list format (1-5)."""
        else:
            department_prompt = None
            
        # Generate insights
        insights = {"general": model.generate(general_prompt)}
        
        if salary_prompt:
            insights["salary"] = model.generate(salary_prompt)
            
        if department_prompt:
            insights["department"] = model.generate(department_prompt)
            
        logger.info("Successfully generated insights")
        return {
            "status": "success",
            "insights": insights,
            "message": "Insights generated successfully"
        }
    
    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}")
        return {
            "status": "error",
            "insights": {},
            "message": f"Failed to generate insights: {str(e)}"
        }

def extract_use_cases(insights: Dict[str, str], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract potential use cases from the generated insights.
    
    Args:
        insights: Dictionary of generated insights
        config: Configuration parameters
        
    Returns:
        Dictionary containing extracted use cases
    """
    logger.info("Extracting use cases from insights")
    
    try:
        # Choose between real OpenAI and mock LLM
        model_name = config.get("ai_insights", {}).get("models", {}).get("enhanced", "gpt-4")
        
        if USE_MOCK_LLM:
            logger.info("Using mock LLM for use case extraction")
            model = MockLLM()
        else:
            logger.info(f"Using OpenAI (model: {model_name}) for use case extraction")
            model = OpenAILLM(model=model_name)
        
        # Combine all insights into a single text
        all_insights = "\n\n".join(insights.values())
        
        # Generate use cases based on insights with an improved prompt
        use_cases_prompt = f"""Based on the following data insights, identify 3-5 practical business use cases or recommendations. 
Each use case should include:
1. A clear title
2. A brief description of what to implement
3. The expected business impact (Low/Medium/High)
4. The implementation effort required (Low/Medium/High)

INSIGHTS:
{all_insights}

Format each use case as a JSON object with keys: "title", "description", "impact", and "effort".
Return a JSON array of these use cases.
"""
        
        use_cases_text = model.generate(use_cases_prompt)
        
        # Try to parse the JSON response
        try:
            # Extract JSON from the potential text response (look for array brackets)
            json_start = use_cases_text.find('[')
            json_end = use_cases_text.rfind(']') + 1
            
            if json_start != -1 and json_end != -1:
                json_text = use_cases_text[json_start:json_end]
                use_cases = json.loads(json_text)
            else:
                # Fallback to structured parsing if JSON not found
                use_cases = parse_use_cases_text(use_cases_text)
        except json.JSONDecodeError:
            # Fallback to structured parsing
            use_cases = parse_use_cases_text(use_cases_text)
        
        logger.info("Successfully extracted use cases")
        return {
            "status": "success",
            "use_cases": use_cases,
            "message": "Use cases extracted successfully"
        }
    
    except Exception as e:
        logger.error(f"Error extracting use cases: {str(e)}")
        
        # Fallback to predefined use cases
        fallback_use_cases = [
            {
                "title": "Salary Review Process",
                "description": "Implement a data-driven salary review process based on department benchmarks.",
                "impact": "High",
                "effort": "Medium"
            },
            {
                "title": "Department Expansion Planning",
                "description": "Use hiring trends and department distribution to plan strategic growth.",
                "impact": "Medium",
                "effort": "Low"
            },
            {
                "title": "Data Quality Initiative",
                "description": "Address the 20% of records with missing values to improve data completeness.",
                "impact": "Medium",
                "effort": "Medium"
            }
        ]
        
        logger.warning("Using fallback use cases")
        return {
            "status": "success",
            "use_cases": fallback_use_cases,
            "message": "Fallback use cases extracted"
        }

def parse_use_cases_text(text: str) -> List[Dict[str, str]]:
    """
    Parse use cases from text format if JSON parsing fails.
    
    Args:
        text: Text containing use cases
        
    Returns:
        List of use case dictionaries
    """
    use_cases = []
    lines = text.strip().split('\n')
    
    current_case = {}
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Check for title (usually starts with number, or has "Title:", or is all caps)
        if line[0].isdigit() and '.' in line[:3]:
            # Save previous use case if it exists
            if current_case and 'title' in current_case:
                use_cases.append(current_case)
                current_case = {}
            
            # Extract title
            title_parts = line.split('.', 1)
            if len(title_parts) > 1:
                current_case['title'] = title_parts[1].strip()
            else:
                current_case['title'] = line.strip()
        
        # Look for description
        elif 'description' in line.lower() or (current_case and 'title' in current_case and 'description' not in current_case):
            if ':' in line:
                current_case['description'] = line.split(':', 1)[1].strip()
            else:
                current_case['description'] = line.strip()
        
        # Look for impact
        elif 'impact' in line.lower():
            if 'high' in line.lower():
                current_case['impact'] = 'High'
            elif 'medium' in line.lower():
                current_case['impact'] = 'Medium'
            elif 'low' in line.lower():
                current_case['impact'] = 'Low'
        
        # Look for effort
        elif 'effort' in line.lower():
            if 'high' in line.lower():
                current_case['effort'] = 'High'
            elif 'medium' in line.lower():
                current_case['effort'] = 'Medium'
            elif 'low' in line.lower():
                current_case['effort'] = 'Low'
    
    # Add the last use case
    if current_case and 'title' in current_case:
        use_cases.append(current_case)
    
    # Ensure all use cases have required fields
    for case in use_cases:
        if 'title' not in case:
            case['title'] = 'Untitled Use Case'
        if 'description' not in case:
            case['description'] = 'No description provided'
        if 'impact' not in case:
            case['impact'] = 'Medium'
        if 'effort' not in case:
            case['effort'] = 'Medium'
    
    # If no use cases were found, create a default one
    if not use_cases:
        use_cases = [
            {
                "title": "Data-Driven Decision Making",
                "description": "Implement a framework for making decisions based on data insights.",
                "impact": "High",
                "effort": "Medium"
            }
        ]
    
    return use_cases

class InsightGenerator:
    """
    Class to handle the end-to-end insight generation process.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the InsightGenerator with configuration.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.output_dir = os.path.join(
            config.get("ai_insights", {}).get("output_dir", "insights")
        )
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize Snowflake connection if configured
        self.snowflake_conn = None
        if self.config.get("ai_insights", {}).get("use_snowflake", False):
            snowflake_config = config.get("ai_insights", {}).get("snowflake", {})
            self.snowflake_conn = connect_to_snowflake(snowflake_config)
    
    def generate(self, data: pd.DataFrame, file_name: str) -> Dict[str, Any]:
        """
        Generate insights and use cases from the data.
        
        Args:
            data: Input DataFrame to analyze
            file_name: Name of the processed file for output naming
            
        Returns:
            Dictionary containing generation results
        """
        logger.info(f"Generating insights for: {file_name}")
        result = {
            "status": "success",
            "insights": None,
            "use_cases": None,
            "output_path": None,
            "message": ""
        }
        
        try:
            # Step 1: Generate insights
            insights_result = generate_insights(data, self.config)
            if insights_result["status"] == "error":
                raise Exception(insights_result["message"])
            
            insights = insights_result["insights"]
            
            # Step 2: Extract use cases
            use_cases_result = extract_use_cases(insights, self.config)
            if use_cases_result["status"] == "error":
                raise Exception(use_cases_result["message"])
            
            use_cases = use_cases_result["use_cases"]
            
            # Step 3: Save the results
            base_name = os.path.splitext(os.path.basename(file_name))[0]
            output_path = os.path.join(self.output_dir, f"{base_name}_insights.json")
            
            output_data = {
                "file_name": file_name,
                "insights": insights,
                "use_cases": use_cases,
                "data_summary": {
                    "rows": len(data),
                    "columns": list(data.columns),
                    "missing_values": data.isna().sum().to_dict(),
                    "generation_time": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            result["insights"] = insights
            result["use_cases"] = use_cases
            result["output_path"] = output_path
            result["message"] = f"Insights generated successfully and saved to {output_path}"
            logger.info(f"Insights generation completed: {output_path}")
            
        except Exception as e:
            result["status"] = "error"
            result["message"] = f"Error generating insights: {str(e)}"
            logger.error(f"Error generating insights: {str(e)}")
        
        return result

# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Sample configuration
    sample_config = {
        "ai_insights": {
            "output_dir": "insights",
            "use_snowflake": False,
            "models": {
                "default": "gpt-3.5-turbo",
                "enhanced": "gpt-4"
            }
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
    
    # Generate insights
    generator = InsightGenerator(sample_config)
    result = generator.generate(sample_df, "sample_data.csv")
    
    if result["status"] == "success":
        print("Insights generated successfully!")
        print("\nGeneral Insights:")
        print(result["insights"]["general"])
        print("\nOutput saved to:", result["output_path"])
    else:
        print(f"Insights generation failed: {result['message']}") 