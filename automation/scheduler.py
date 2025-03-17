import os
import json
import logging
import datetime
from typing import Dict, Any, List, Optional
import sys
import tempfile
import shutil
import traceback

# Configure logging
logger = logging.getLogger(__name__)

# Mock Airflow classes for demonstration purposes
# In a real implementation, these would be actual Airflow imports
class MockDAG:
    def __init__(self, dag_id, default_args, schedule_interval, catchup=False):
        self.dag_id = dag_id
        self.default_args = default_args
        self.schedule_interval = schedule_interval
        self.catchup = catchup
        self.tasks = []
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class MockPythonOperator:
    def __init__(self, task_id, python_callable, op_kwargs, dag):
        self.task_id = task_id
        self.python_callable = python_callable
        self.op_kwargs = op_kwargs
        self.dag = dag
        dag.tasks.append(self)
    
    def set_upstream(self, upstream_task):
        pass
    
    def set_downstream(self, downstream_task):
        pass

# Add parent directory to path to allow imports from other modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Add project root to sys.path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Import project modules
from data_ingestion.file_handler import FileHandler
from data_processing.data_processor import DataProcessor
from ai_insights.insight_generator import InsightGenerator
from reporting.report_generator import ReportGenerator
from config.config_loader import load_config

# Ensure logs directory exists
os.makedirs(os.path.join(project_root, 'logs'), exist_ok=True)

# Define Airflow DAG configurations
default_args = {
    'owner': 'ai_data_automation',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
}

def create_analysis_pipeline_dag(dag_id, default_args, schedule_interval, config):
    """
    Create an Airflow DAG for the data analysis pipeline.
    
    Args:
        dag_id: DAG identifier
        default_args: Default DAG arguments
        schedule_interval: Schedule interval for the DAG
        config: Configuration dictionary
        
    Returns:
        Airflow DAG instance
    """
    try:
        from airflow import DAG
        from airflow.operators.python import PythonOperator
        from airflow.operators.bash import BashOperator
        from airflow.sensors.filesystem import FileSensor
        from airflow.utils.dates import days_ago
        
        # Create DAG
        dag = DAG(
            dag_id,
            default_args=default_args,
            description='Data Analysis Automation Pipeline',
            schedule_interval=schedule_interval,
            catchup=False,
        )
        
        # Define tasks
        
        # Task to check source data directory for new files
        check_source_dir = BashOperator(
            task_id='check_source_dir',
            bash_command=f'ls -la {os.path.join(project_root, "source_data")} || true',
            dag=dag,
        )
        
        # Function to find new files in source directory
        def find_new_files(**kwargs):
            source_dir = os.path.join(project_root, 'source_data')
            processed_dir = os.path.join(project_root, 'processed_data')
            
            # Get list of files in source directory
            source_files = []
            try:
                source_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
            except Exception as e:
                logger.error(f"Error listing source directory: {str(e)}")
                return []
            
            # Get list of already processed files
            processed_files = []
            try:
                processed_files = [os.path.splitext(f)[0] for f in os.listdir(processed_dir) 
                                  if os.path.isfile(os.path.join(processed_dir, f))]
            except Exception as e:
                logger.error(f"Error listing processed directory: {str(e)}")
                
            # Filter for new files
            new_files = []
            for file in source_files:
                base_name = os.path.splitext(file)[0]
                if base_name not in processed_files:
                    file_path = os.path.join(source_dir, file)
                    file_handler = FileHandler(config)
                    if file_handler.is_supported_file_type(file_path):
                        new_files.append(file_path)
            
            return new_files
        
        discover_new_files = PythonOperator(
            task_id='discover_new_files',
            python_callable=find_new_files,
            provide_context=True,
            dag=dag,
        )
        
        # Function to process a single file
        def process_file(file_path, **kwargs):
            try:
                logger.info(f"Processing file: {file_path}")
                
                # Step 1: Load file
                file_handler = FileHandler(config)
                file_info = file_handler.load_file(file_path)
                if file_info["status"] == "error":
                    raise Exception(file_info["message"])
                
                data = file_info["data"]
                
                # Step 2: Process data
                data_processor = DataProcessor(config)
                processed_data = data_processor.process(data, file_path)
                if processed_data["status"] == "error":
                    raise Exception(processed_data["message"])
                
                data = processed_data["data"]
                
                # Step 3: Generate insights
                insight_generator = InsightGenerator(config)
                insights_result = insight_generator.generate(data, file_path)
                if insights_result["status"] == "error":
                    raise Exception(insights_result["message"])
                
                # Step 4: Generate report
                report_generator = ReportGenerator(config)
                report_result = report_generator.generate_report(
                    data, 
                    insights_result["output_path"],
                    file_path
                )
                
                if report_result["status"] == "error":
                    raise Exception(report_result["message"])
                
                logger.info(f"Successfully processed file: {file_path}")
                return {
                    "status": "success",
                    "file_path": file_path,
                    "processed_data_path": processed_data["output_path"],
                    "insights_path": insights_result["output_path"],
                    "report_paths": report_result["report_paths"]
                }
                
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                logger.error(traceback.format_exc())
                return {
                    "status": "error",
                    "file_path": file_path,
                    "message": str(e)
                }
        
        # Function to process all new files
        def process_files(**kwargs):
            ti = kwargs['ti']
            new_files = ti.xcom_pull(task_ids='discover_new_files')
            
            if not new_files:
                logger.info("No new files to process")
                return {
                    "status": "success",
                    "message": "No new files to process",
                    "processed_files": []
                }
            
            logger.info(f"Found {len(new_files)} new files to process")
            results = []
            
            for file_path in new_files:
                result = process_file(file_path)
                results.append(result)
            
            # Count successes and failures
            successes = sum(1 for r in results if r["status"] == "success")
            failures = sum(1 for r in results if r["status"] == "error")
            
            return {
                "status": "completed",
                "message": f"Processed {len(results)} files: {successes} successes, {failures} failures",
                "processed_files": results
            }
        
        process_new_files = PythonOperator(
            task_id='process_new_files',
            python_callable=process_files,
            provide_context=True,
            dag=dag,
        )
        
        # Task to clean up temporary files after successful processing
        def cleanup_temp_files(**kwargs):
            temp_dir = os.path.join(project_root, 'temp')
            if os.path.exists(temp_dir):
                try:
                    # Delete files older than 24 hours
                    cutoff_time = datetime.now() - datetime.timedelta(hours=24)
                    for filename in os.listdir(temp_dir):
                        file_path = os.path.join(temp_dir, filename)
                        if os.path.isfile(file_path):
                            file_mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                            if file_mod_time < cutoff_time:
                                os.remove(file_path)
                                logger.info(f"Deleted old temp file: {file_path}")
                    
                    logger.info("Temporary file cleanup completed")
                except Exception as e:
                    logger.error(f"Error cleaning up temp files: {str(e)}")
            
            return {"status": "success", "message": "Cleanup completed"}
        
        cleanup = PythonOperator(
            task_id='cleanup_temp_files',
            python_callable=cleanup_temp_files,
            provide_context=True,
            dag=dag,
        )
        
        # Set up task dependencies
        check_source_dir >> discover_new_files >> process_new_files >> cleanup
        
        return dag
        
    except ImportError:
        logger.warning("Airflow not installed. Cannot create DAG.")
        return None

def run_scheduled_batch(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a scheduled batch processing job without Airflow.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with batch processing results
    """
    logger.info("Starting scheduled batch processing")
    source_dir = os.path.join(project_root, 'source_data')
    processed_marker_dir = os.path.join(project_root, 'processed_data')
    
    # Create necessary directories
    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(processed_marker_dir, exist_ok=True)
    
    # Get list of files in source directory
    source_files = []
    try:
        source_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    except Exception as e:
        logger.error(f"Error listing source directory: {str(e)}")
        return {
            "status": "error",
            "message": f"Error listing source directory: {str(e)}",
            "processed_files": []
        }
    
    # Get list of already processed files
    processed_files = []
    try:
        processed_files = [os.path.splitext(f)[0] for f in os.listdir(processed_marker_dir) 
                          if os.path.isfile(os.path.join(processed_marker_dir, f))]
    except Exception as e:
        logger.error(f"Error listing processed directory: {str(e)}")
    
    # Filter for new files
    new_files = []
    file_handler = FileHandler(config)
    for file in source_files:
        base_name = os.path.splitext(file)[0]
        if base_name not in processed_files:
            file_path = os.path.join(source_dir, file)
            if file_handler.is_supported_file_type(file_path):
                new_files.append(file_path)
    
    if not new_files:
        logger.info("No new files to process")
        return {
            "status": "success",
            "message": "No new files to process",
            "processed_files": []
        }
    
    logger.info(f"Found {len(new_files)} new files to process")
    results = []
    
    for file_path in new_files:
        try:
            logger.info(f"Processing file: {file_path}")
            
            # Step 1: Load file
            file_info = file_handler.load_file(file_path)
            if file_info["status"] == "error":
                logger.error(f"Error loading file {file_path}: {file_info['message']}")
                results.append({
                    "status": "error",
                    "file_path": file_path,
                    "message": file_info["message"]
                })
                continue
            
            data = file_info["data"]
            
            # Step 2: Process data
            data_processor = DataProcessor(config)
            processed_data = data_processor.process(data, file_path)
            if processed_data["status"] == "error":
                logger.error(f"Error processing file {file_path}: {processed_data['message']}")
                results.append({
                    "status": "error",
                    "file_path": file_path,
                    "message": processed_data["message"]
                })
                continue
            
            data = processed_data["data"]
            
            # Step 3: Generate insights
            insight_generator = InsightGenerator(config)
            insights_result = insight_generator.generate(data, file_path)
            if insights_result["status"] == "error":
                logger.error(f"Error generating insights for {file_path}: {insights_result['message']}")
                results.append({
                    "status": "error",
                    "file_path": file_path,
                    "message": insights_result["message"]
                })
                continue
            
            # Step 4: Generate report
            report_generator = ReportGenerator(config)
            report_result = report_generator.generate_report(
                data, 
                insights_result["output_path"],
                file_path
            )
            
            if report_result["status"] == "error":
                logger.error(f"Error generating report for {file_path}: {report_result['message']}")
                results.append({
                    "status": "error",
                    "file_path": file_path,
                    "message": report_result["message"]
                })
                continue
            
            # Mark file as processed by creating a marker file
            base_name = os.path.basename(file_path)
            marker_path = os.path.join(processed_marker_dir, f"{os.path.splitext(base_name)[0]}.processed")
            with open(marker_path, 'w') as f:
                json.dump({
                    "file_path": file_path,
                    "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "processed_data_path": processed_data["output_path"],
                    "insights_path": insights_result["output_path"],
                    "report_paths": report_result["report_paths"]
                }, f, indent=2)
            
            logger.info(f"Successfully processed file: {file_path}")
            results.append({
                "status": "success",
                "file_path": file_path,
                "processed_data_path": processed_data["output_path"],
                "insights_path": insights_result["output_path"],
                "report_paths": report_result["report_paths"]
            })
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            logger.error(traceback.format_exc())
            results.append({
                "status": "error",
                "file_path": file_path,
                "message": str(e)
            })
    
    # Count successes and failures
    successes = sum(1 for r in results if r["status"] == "success")
    failures = sum(1 for r in results if r["status"] == "error")
    
    logger.info(f"Batch processing completed: {successes} successes, {failures} failures")
    return {
        "status": "completed",
        "message": f"Processed {len(results)} files: {successes} successes, {failures} failures",
        "processed_files": results
    }

def get_airflow_dag(schedule_interval: str = "0 0 * * *"):
    """
    Get Airflow DAG for scheduled processing.
    
    Args:
        schedule_interval: Cron expression for scheduling
        
    Returns:
        Airflow DAG if Airflow is installed, None otherwise
    """
    try:
        config_path = os.path.join(project_root, 'config', 'config.json')
        config = load_config(config_path)
        
        return create_analysis_pipeline_dag(
            'ai_data_analysis_automation',
            default_args,
            schedule_interval,
            config
        )
    except Exception as e:
        logger.error(f"Error creating Airflow DAG: {str(e)}")
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs(os.path.join(project_root, 'source_data'), exist_ok=True)
    os.makedirs(os.path.join(project_root, 'processed_data'), exist_ok=True)
    os.makedirs(os.path.join(project_root, 'insights'), exist_ok=True)
    os.makedirs(os.path.join(project_root, 'reports'), exist_ok=True)
    os.makedirs(os.path.join(project_root, 'temp'), exist_ok=True)
    
    # Load configuration
    try:
        config_path = os.path.join(project_root, 'config', 'config.json')
        config = load_config(config_path)
        
        # Run scheduled batch without Airflow
        result = run_scheduled_batch(config)
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        logger.error(f"Error running scheduler: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"Error: {str(e)}")
        sys.exit(1) 