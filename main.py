"""
FastAPI Backend for AutoPRISMA System
Run with: python main.py
Or: uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
import asyncio
from datetime import datetime
from pathlib import Path
import uuid

from config import settings
from orchestrator import AutoPRISMAOrchestrator
from state import Document, ReviewReport

# Set up logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AutoPRISMA API",
    description="Automated PRISMA 2020 Systematic Literature Review System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for review jobs (in production, use a database)
review_jobs: Dict[str, Dict[str, Any]] = {}


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ReviewRequest(BaseModel):
    """Request model for starting a systematic review."""
    research_question: str = Field(
        description="The research question for the systematic review",
        min_length=10
    )
    databases: List[str] = Field(
        default=["semantic_scholar", "arxiv"],
        description="Databases to search"
    )
    date_from: Optional[str] = Field(
        None,
        description="Start date in YYYY-MM-DD format"
    )
    date_to: Optional[str] = Field(
        None,
        description="End date in YYYY-MM-DD format"
    )
    enable_hitl: bool = Field(
        default=False,
        description="Enable human-in-the-loop checkpoints"
    )


class ReviewStatusResponse(BaseModel):
    """Response model for review status."""
    job_id: str
    status: str  # "pending", "running", "completed", "failed"
    current_stage: Optional[str] = None
    progress: float  # 0.0 to 1.0
    started_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class ReviewResultResponse(BaseModel):
    """Response model for completed review results."""
    job_id: str
    status: str
    research_question: str
    summary: Dict[str, Any]
    report_path: Optional[str] = None
    audit_trail_path: Optional[str] = None
    diagram_path: Optional[str] = None


# ============================================================================
# BACKGROUND TASK
# ============================================================================

async def run_review_background(job_id: str, request: ReviewRequest):
    """Background task to run the review."""
    try:
        logger.info(f"Starting review job {job_id}")
        review_jobs[job_id]["status"] = "running"
        review_jobs[job_id]["current_stage"] = "initializing"
        
        # Prepare date range
        date_range = None
        if request.date_from and request.date_to:
            date_range = (request.date_from, request.date_to)
        
        # Run orchestrator
        orchestrator = AutoPRISMAOrchestrator()
        result = await orchestrator.run_review(
            research_question=request.research_question,
            databases=request.databases,
            date_range=date_range,
            enable_hitl=request.enable_hitl
        )
        
        # Update job status
        if result["status"] == "success":
            review_jobs[job_id]["status"] = "completed"
            review_jobs[job_id]["result"] = result
            review_jobs[job_id]["completed_at"] = datetime.now()
            logger.info(f"Review job {job_id} completed successfully")
        else:
            review_jobs[job_id]["status"] = "failed"
            review_jobs[job_id]["error_message"] = result.get("error", "Unknown error")
            logger.error(f"Review job {job_id} failed: {result.get('error')}")
    
    except Exception as e:
        logger.error(f"Review job {job_id} failed with exception: {e}", exc_info=True)
        review_jobs[job_id]["status"] = "failed"
        review_jobs[job_id]["error_message"] = str(e)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "AutoPRISMA API",
        "version": "1.0.0",
        "description": "Automated PRISMA 2020 Systematic Literature Review System",
        "docs": "/docs",
        "status": "operational"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/reviews", response_model=Dict[str, str])
async def create_review(request: ReviewRequest, background_tasks: BackgroundTasks):
    """
    Start a new systematic review.
    
    The review runs in the background. Use the job_id to check status.
    """
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Initialize job record
    review_jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "research_question": request.research_question,
        "started_at": datetime.now(),
        "progress": 0.0,
        "request": request.model_dump()
    }
    
    # Start background task
    background_tasks.add_task(run_review_background, job_id, request)
    
    logger.info(f"Created review job {job_id} for question: {request.research_question}")
    
    return {
        "job_id": job_id,
        "message": "Review started",
        "status_url": f"/reviews/{job_id}/status"
    }


@app.get("/reviews/{job_id}/status", response_model=ReviewStatusResponse)
async def get_review_status(job_id: str):
    """Get the status of a review job."""
    if job_id not in review_jobs:
        raise HTTPException(status_code=404, detail="Review job not found")
    
    job = review_jobs[job_id]
    
    # Calculate progress based on stage
    stage_progress = {
        "initializing": 0.0,
        "query_strategy_complete": 0.15,
        "literature_retrieval_complete": 0.35,
        "screening_criteria_complete": 0.45,
        "abstract_evaluation_complete": 0.65,
        "synthesis_complete": 0.85,
        "report_generation_complete": 1.0
    }
    
    current_stage = job.get("current_stage", "initializing")
    progress = stage_progress.get(current_stage, 0.0)
    
    if job["status"] == "completed":
        progress = 1.0
    elif job["status"] == "failed":
        progress = job.get("progress", 0.0)
    
    return ReviewStatusResponse(
        job_id=job_id,
        status=job["status"],
        current_stage=current_stage,
        progress=progress,
        started_at=job["started_at"],
        completed_at=job.get("completed_at"),
        error_message=job.get("error_message")
    )


@app.get("/reviews/{job_id}/result")
async def get_review_result(job_id: str):
    """Get the full results of a completed review."""
    if job_id not in review_jobs:
        raise HTTPException(status_code=404, detail="Review job not found")
    
    job = review_jobs[job_id]
    
    if job["status"] not in ["completed", "failed"]:
        raise HTTPException(
            status_code=400,
            detail=f"Review is not complete yet. Current status: {job['status']}"
        )
    
    if job["status"] == "failed":
        raise HTTPException(
            status_code=500,
            detail=f"Review failed: {job.get('error_message', 'Unknown error')}"
        )
    
    result = job.get("result", {})
    final_state = result.get("final_state", {})
    final_report = final_state.get("final_report")
    
    # Prepare summary
    summary = {
        "total_papers_retrieved": len(final_state.get("retrieved_documents", [])),
        "papers_included": len(final_state.get("included_documents", [])),
        "papers_excluded": len(final_state.get("excluded_documents", [])),
        "papers_borderline": len(final_state.get("borderline_documents", [])),
        "themes_identified": len(final_state.get("synthesis_result", {}).get("themes", [])) if final_state.get("synthesis_result") else 0,
        "research_gaps": len(final_state.get("synthesis_result", {}).get("research_gaps", [])) if final_state.get("synthesis_result") else 0
    }
    
    return ReviewResultResponse(
        job_id=job_id,
        status=job["status"],
        research_question=job["research_question"],
        summary=summary,
        report_path=final_report.report_path if final_report else None,
        audit_trail_path=result.get("audit_trail_path"),
        diagram_path=final_report.metadata.get("diagram_path") if final_report else None
    )


@app.get("/reviews/{job_id}/report")
async def download_report(job_id: str, format: str = "markdown"):
    """Download the generated report in specified format."""
    if job_id not in review_jobs:
        raise HTTPException(status_code=404, detail="Review job not found")
    
    job = review_jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Review not completed yet")
    
    result = job.get("result", {})
    final_state = result.get("final_state", {})
    final_report = final_state.get("final_report")
    
    if not final_report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    # Get report path for requested format
    report_paths = final_report.metadata.get("report_paths", {})
    file_path = report_paths.get(format.lower())
    
    if not file_path or not Path(file_path).exists():
        raise HTTPException(
            status_code=404,
            detail=f"Report format '{format}' not available. Available: {list(report_paths.keys())}"
        )
    
    return FileResponse(
        path=file_path,
        filename=f"systematic_review_{job_id}.{format}",
        media_type="application/octet-stream"
    )


@app.get("/reviews/{job_id}/audit")
async def download_audit_trail(job_id: str):
    """Download the audit trail HTML report."""
    if job_id not in review_jobs:
        raise HTTPException(status_code=404, detail="Review job not found")
    
    job = review_jobs[job_id]
    
    if job["status"] not in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="Review not completed yet")
    
    result = job.get("result", {})
    audit_path = result.get("audit_trail_path")
    
    if not audit_path or not Path(audit_path).exists():
        raise HTTPException(status_code=404, detail="Audit trail not found")
    
    return FileResponse(
        path=audit_path,
        filename=f"audit_trail_{job_id}.html",
        media_type="text/html"
    )


@app.get("/reviews")
async def list_reviews():
    """List all review jobs."""
    jobs_list = [
        {
            "job_id": job_id,
            "status": job["status"],
            "research_question": job["research_question"],
            "started_at": job["started_at"].isoformat(),
            "completed_at": job.get("completed_at").isoformat() if job.get("completed_at") else None
        }
        for job_id, job in review_jobs.items()
    ]
    
    return {
        "total": len(jobs_list),
        "jobs": jobs_list
    }


@app.delete("/reviews/{job_id}")
async def delete_review(job_id: str):
    """Delete a review job and its results."""
    if job_id not in review_jobs:
        raise HTTPException(status_code=404, detail="Review job not found")
    
    del review_jobs[job_id]
    
    return {"message": f"Review job {job_id} deleted"}


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("="*80)
    logger.info("🚀 Starting AutoPRISMA API Server")
    logger.info("="*80)
    logger.info(f"Host: {settings.api_host}")
    logger.info(f"Port: {settings.api_port}")
    logger.info(f"Docs: http://{settings.api_host}:{settings.api_port}/docs")
    logger.info("="*80)
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower()
    )
