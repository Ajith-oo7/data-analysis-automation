import os
import json
import logging
import traceback
from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Body, Depends, Query, status, Response
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil
import uuid
import sys
import time
from pydantic import BaseModel, Field

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, 'logs', 'api_service.log'), mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Ensure logs directory exists
os.makedirs(os.path.join(project_root, 'logs'), exist_ok=True)

# Create FastAPI app
app = FastAPI(
    title="AI Data Analysis Automation API",
    description="API for automated data analysis with AI-generated insights and reporting",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class AnalysisRequest(BaseModel):
    file_path: str = Field(..., description="Path to the file for analysis")
    mode: str = Field("full", description="Analysis mode: 'full', 'insights_only', or 'process_only'")
    
class AnalysisResponse(BaseModel):
    job_id: str = Field(..., description="Unique ID for the analysis job")
    status: str = Field(..., description="Status of the job (queued, processing, completed, failed)")
    message: str = Field(..., description="Status message")

class JobStatusResponse(BaseModel):
    job_id: str = Field(..., description="Unique ID for the analysis job")
    status: str = Field(..., description="Status of the job (queued, processing, completed, failed)")
    progress: float = Field(..., description="Progress percentage (0-100)")
    created_at: str = Field(..., description="Job creation timestamp")
    completed_at: Optional[str] = Field(None, description="Job completion timestamp")
    result: Optional[Dict[str, Any]] = Field(None, description="Result data if job is completed")
    error: Optional[str] = Field(None, description="Error message if job failed")

# In-memory job storage (in production, use a database)
jobs = {}

def get_config():
    """Dependency to get configuration."""
    try:
        config_path = os.path.join(project_root, 'config', 'config.json')
        config = load_config(config_path)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load application configuration"
        )

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint returning API information."""
    return {
        "message": "AI Data Analysis Automation API",
        "version": "1.0.0",
        "documentation": "/docs"
    }

@app.post("/api/upload", tags=["Data Ingestion"], response_model=AnalysisResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    mode: str = Query("full", description="Analysis mode: 'full', 'insights_only', or 'process_only'"),
    config: Dict[str, Any] = Depends(get_config)
):
    """
    Upload a file for analysis.
    
    - **file**: The file to analyze
    - **mode**: Analysis mode (full, insights_only, process_only)
    
    Returns a job ID for tracking the analysis
    """
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided"
        )
    
    try:
        # Create upload directory if it doesn't exist
        upload_dir = os.path.join(project_root, 'source_data')
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save uploaded file
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Create job record
        job_id = str(uuid.uuid4())
        jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "file_path": file_path,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "completed_at": None,
            "result": None,
            "error": None
        }
        
        # Start background task for processing
        background_tasks.add_task(
            process_file_task, 
            job_id=job_id, 
            file_path=file_path, 
            mode=mode,
            config=config
        )
        
        logger.info(f"File upload accepted: {file.filename}, Job ID: {job_id}")
        return {
            "job_id": job_id,
            "status": "queued",
            "message": f"File {file.filename} uploaded successfully and queued for processing"
        }
        
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing upload: {str(e)}"
        )

@app.post("/api/analyze", tags=["Analysis"], response_model=AnalysisResponse)
async def analyze_file(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    config: Dict[str, Any] = Depends(get_config)
):
    """
    Analyze a file that already exists in the system.
    
    - **file_path**: Path to the file for analysis
    - **mode**: Analysis mode (full, insights_only, process_only)
    
    Returns a job ID for tracking the analysis
    """
    file_path = request.file_path
    
    # Validate file path
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {file_path}"
        )
    
    try:
        # Create job record
        job_id = str(uuid.uuid4())
        jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "file_path": file_path,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "completed_at": None,
            "result": None,
            "error": None
        }
        
        # Start background task for processing
        background_tasks.add_task(
            process_file_task, 
            job_id=job_id, 
            file_path=file_path, 
            mode=request.mode,
            config=config
        )
        
        logger.info(f"Analysis requested for: {file_path}, Job ID: {job_id}")
        return {
            "job_id": job_id,
            "status": "queued",
            "message": f"File analysis queued for {file_path}"
        }
        
    except Exception as e:
        logger.error(f"Error starting analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting analysis: {str(e)}"
        )

@app.get("/api/jobs/{job_id}", tags=["Jobs"], response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get the status of an analysis job.
    
    - **job_id**: The ID of the job to check
    
    Returns the current status of the job
    """
    if job_id not in jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}"
        )
    
    return jobs[job_id]

@app.get("/api/jobs", tags=["Jobs"], response_model=List[JobStatusResponse])
async def list_jobs():
    """
    List all analysis jobs.
    
    Returns a list of all jobs and their statuses
    """
    return list(jobs.values())

@app.get("/api/reports/{job_id}", tags=["Reports"])
async def get_report(job_id: str):
    """
    Get the HTML report for a completed analysis job.
    
    - **job_id**: The ID of the job
    
    Returns the HTML report as a file download
    """
    if job_id not in jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}"
        )
    
    job = jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job is not completed: {job['status']}"
        )
    
    if "report_path" not in job["result"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No report available for this job"
        )
    
    report_path = job["result"]["report_path"]
    
    if not os.path.exists(report_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report file not found: {report_path}"
        )
    
    return FileResponse(
        path=report_path,
        filename=os.path.basename(report_path),
        media_type="text/html"
    )

@app.get("/api/insights/{job_id}", tags=["Insights"])
async def get_insights(job_id: str):
    """
    Get the JSON insights for a completed analysis job.
    
    - **job_id**: The ID of the job
    
    Returns the insights JSON
    """
    if job_id not in jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}"
        )
    
    job = jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job is not completed: {job['status']}"
        )
    
    if "insights_path" not in job["result"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No insights available for this job"
        )
    
    insights_path = job["result"]["insights_path"]
    
    if not os.path.exists(insights_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Insights file not found: {insights_path}"
        )
    
    # Read and return insights
    with open(insights_path, 'r') as f:
        insights = json.load(f)
    
    return insights

@app.delete("/api/jobs/{job_id}", tags=["Jobs"])
async def delete_job(job_id: str, response: Response):
    """
    Delete an analysis job and its results.
    
    - **job_id**: The ID of the job to delete
    
    Returns a success message
    """
    if job_id not in jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}"
        )
    
    # Remove job from memory
    job = jobs.pop(job_id)
    
    # In a real implementation, you'd also delete associated files
    
    response.status_code = status.HTTP_204_NO_CONTENT
    return None

# Background task for file processing
async def process_file_task(job_id: str, file_path: str, mode: str, config: Dict[str, Any]):
    """Background task to process a file through the analysis pipeline."""
    job = jobs[job_id]
    job["status"] = "processing"
    
    try:
        # Initialize components
        file_handler = FileHandler(config)
        data_processor = DataProcessor(config)
        insight_generator = InsightGenerator(config)
        report_generator = ReportGenerator(config)
        
        # Step 1: Load and validate file (25%)
        job["progress"] = 5
        logger.info(f"Job {job_id}: Loading file {file_path}")
        
        file_info = file_handler.load_file(file_path)
        if file_info["status"] == "error":
            raise Exception(file_info["message"])
        
        data = file_info["data"]
        job["progress"] = 25
        
        result = {"file_path": file_path}
        
        # Step 2: Process data (50%)
        if mode in ["full", "process_only"]:
            logger.info(f"Job {job_id}: Processing data")
            processed_data = data_processor.process(data, file_path)
            if processed_data["status"] == "error":
                raise Exception(processed_data["message"])
            
            data = processed_data["data"]
            result["processed_data_path"] = processed_data["output_path"]
            job["progress"] = 50
        
        # Step 3: Generate insights (75%)
        if mode in ["full", "insights_only"]:
            logger.info(f"Job {job_id}: Generating insights")
            insights_result = insight_generator.generate(data, file_path)
            if insights_result["status"] == "error":
                raise Exception(insights_result["message"])
            
            result["insights_path"] = insights_result["output_path"]
            result["insights"] = insights_result["insights"]
            result["use_cases"] = insights_result["use_cases"]
            job["progress"] = 75
        
        # Step 4: Generate report (100%)
        if mode == "full" and "insights_path" in result:
            logger.info(f"Job {job_id}: Generating report")
            report_result = report_generator.generate_report(
                data, 
                result["insights_path"],
                file_path
            )
            
            if report_result["status"] == "error":
                raise Exception(report_result["message"])
            
            result["report_path"] = report_result["report_paths"][0] if report_result["report_paths"] else None
        
        # Mark job as completed
        job["status"] = "completed"
        job["progress"] = 100
        job["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        job["result"] = result
        
        logger.info(f"Job {job_id}: Processing completed successfully")
        
    except Exception as e:
        logger.error(f"Job {job_id}: Processing failed: {str(e)}")
        job["status"] = "failed"
        job["error"] = str(e)
        job["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

# Mount static files directory for reports
try:
    reports_dir = os.path.join(project_root, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    app.mount("/reports", StaticFiles(directory=reports_dir), name="reports")
except Exception as e:
    logger.error(f"Error mounting static files directory: {str(e)}")

# Main function to run the API server
def start_api_server(host="0.0.0.0", port=8000):
    """Start the FastAPI server."""
    import uvicorn
    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs(os.path.join(project_root, 'source_data'), exist_ok=True)
    os.makedirs(os.path.join(project_root, 'processed_data'), exist_ok=True)
    os.makedirs(os.path.join(project_root, 'insights'), exist_ok=True)
    os.makedirs(os.path.join(project_root, 'reports'), exist_ok=True)
    
    # Start API server
    start_api_server() 