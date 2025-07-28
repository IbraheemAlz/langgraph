#!/usr/bin/env python3
"""
RunPod FastAPI Application for Multi-Agent AI Hiring System
Optimized for high-performance processing on RunPod infrastructure
"""

import asyncio
import logging
import time
import os
import sys
import psutil
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, List, Any
import uvicorn

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.config import Config
from src.agents.job_matching_agent import JobMatchingAgent
from src.agents.bias_classification_agent import BiasClassificationAgent

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Config.LOG_FILE_PATH) if Config.LOG_TO_FILE else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Hiring System - RunPod Deployment",
    description="Multi-Agent AI Hiring System optimized for RunPod with local Ollama",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for RunPod access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for agents
job_agent = None
bias_agent = None
startup_time = None

@app.on_event("startup")
async def startup_event():
    """Initialize agents and services on startup"""
    global job_agent, bias_agent, startup_time
    
    startup_time = time.time()
    
    logger.info("🚀 Starting AI Hiring System on RunPod")
    logger.info(f"📍 Pod ID: {Config.RUNPOD_POD_ID}")
    logger.info(f"🤖 Model: {Config.MODEL_NAME}")
    logger.info(f"💾 Workspace: {Config.WORKSPACE_PATH}")
    logger.info(f"⚡ Max Workers: {Config.MAX_WORKERS}")
    logger.info(f"📦 Batch Size: {Config.BATCH_SIZE}")
    
    try:
        # Wait for Ollama to be ready
        await wait_for_ollama()
        
        # Initialize agents
        logger.info("🔧 Initializing agents...")
        job_agent = JobMatchingAgent()
        bias_agent = BiasClassificationAgent()
        
        # Ensure results directory exists
        Path(Config.RESULTS_FOLDER).mkdir(parents=True, exist_ok=True)
        Path(f"{Config.RESULTS_FOLDER}/json").mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ All systems initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

async def wait_for_ollama(max_attempts=30):
    """Wait for Ollama to be ready with detailed status"""
    import requests
    
    logger.info("⏳ Waiting for Ollama service...")
    
    for attempt in range(max_attempts):
        try:
            # Check if Ollama is responding
            response = requests.get(f"{Config.OLLAMA_BASE_URL}/api/version", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Ollama service is ready")
                
                # Check if model is available
                models_response = requests.get(f"{Config.OLLAMA_BASE_URL}/api/tags", timeout=5)
                if models_response.status_code == 200:
                    models = models_response.json().get('models', [])
                    model_names = [model.get('name', '') for model in models]
                    
                    if any(Config.MODEL_NAME in name for name in model_names):
                        logger.info(f"✅ Model {Config.MODEL_NAME} is available")
                        return
                    else:
                        logger.warning(f"⚠️ Model {Config.MODEL_NAME} not found. Available: {model_names}")
                        if attempt < max_attempts - 1:
                            logger.info("🔄 Model may still be downloading...")
                        
        except Exception as e:
            pass
        
        if attempt < max_attempts - 1:
            logger.info(f"⏳ Waiting for Ollama... (attempt {attempt + 1}/{max_attempts})")
            await asyncio.sleep(2)
        else:
            logger.error("❌ Ollama not ready - continuing anyway")

@app.get("/")
async def root():
    """Root endpoint with system status"""
    uptime = time.time() - startup_time if startup_time else 0
    
    return {
        "message": "AI Hiring System - RunPod Deployment",
        "status": "running",
        "model": Config.MODEL_NAME,
        "pod_id": Config.RUNPOD_POD_ID,
        "workspace": Config.WORKSPACE_PATH,
        "uptime_seconds": uptime,
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze_candidate", 
            "batch": "/batch_analyze",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        import requests
        
        # Check Ollama
        ollama_healthy = False
        ollama_error = None
        try:
            ollama_response = requests.get(f"{Config.OLLAMA_BASE_URL}/api/version", timeout=5)
            ollama_healthy = ollama_response.status_code == 200
        except Exception as e:
            ollama_error = str(e)
        
        # System metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # GPU info (if available)
        gpu_info = None
        try:
            import subprocess
            gpu_result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], 
                                       capture_output=True, text=True, timeout=5)
            if gpu_result.returncode == 0:
                gpu_data = gpu_result.stdout.strip().split(', ')
                gpu_info = {
                    "memory_used_mb": int(gpu_data[0]),
                    "memory_total_mb": int(gpu_data[1]),
                    "utilization_percent": int(gpu_data[2])
                }
        except:
            pass
        
        health_status = {
            "status": "healthy" if ollama_healthy and job_agent and bias_agent else "degraded",
            "timestamp": time.time(),
            "uptime_seconds": time.time() - startup_time if startup_time else 0,
            "services": {
                "ollama": {
                    "status": "healthy" if ollama_healthy else "unhealthy",
                    "error": ollama_error
                },
                "job_agent": "ready" if job_agent else "not_initialized",
                "bias_agent": "ready" if bias_agent else "not_initialized"
            },
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "gpu": gpu_info
            },
            "config": {
                "model": Config.MODEL_NAME,
                "batch_size": Config.BATCH_SIZE,
                "max_workers": Config.MAX_WORKERS,
                "pod_id": Config.RUNPOD_POD_ID
            }
        }
        
        return health_status
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }

@app.post("/analyze_candidate")
async def analyze_candidate(request: Dict[str, Any]):
    """Analyze a single candidate with job matching and bias detection"""
    
    if not job_agent or not bias_agent:
        raise HTTPException(status_code=503, detail="Agents not initialized")
    
    try:
        candidate_data = request.get('candidate_data', {})
        job_requirements = request.get('job_requirements', {})
        
        # Validate required fields
        required_fields = ['Resume', 'Job_Description', 'Transcript', 'Role']
        for field in required_fields:
            if field not in candidate_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Job matching analysis
        start_time = time.time()
        job_analysis = job_agent.run(
            Resume=candidate_data['Resume'],
            Job_Description=candidate_data['Job_Description'],
            Transcript=candidate_data['Transcript'],
            Role=candidate_data['Role']
        )
        job_time = time.time() - start_time
        
        # Bias analysis
        bias_start = time.time()
        bias_analysis = bias_agent.run(
            Resume=candidate_data['Resume'],
            Job_Description=candidate_data['Job_Description'],
            Transcript=candidate_data['Transcript'],
            decision=job_analysis.get('decision', 'unknown'),
            Role=candidate_data['Role'],
            primary_reason=job_analysis.get('primary_reason', '')
        )
        bias_time = time.time() - bias_start
        
        total_time = time.time() - start_time
        
        # Extract key data
        candidate_id = candidate_data.get('id', 'unknown')
        final_decision = job_analysis.get('decision', 'reject')
        bias_classification = bias_analysis.get('classification', 'unbiased')
        primary_reason = job_analysis.get('primary_reason', 'No reason provided')
        specific_feedback = bias_analysis.get('specific_feedback', '')
        
        # Debug: Log candidate ID processing
        if candidate_id == 'unknown':
            logger.warning(f"⚠️ No ID found in candidate_data: {list(candidate_data.keys())}")
        else:
            logger.debug(f"✅ Processing candidate ID: {candidate_id}")
        
        # Build evaluation insights structure
        evaluation_insight = {
            "evaluation_number": 1,
            "decision": final_decision,
            "primary_reason": primary_reason,
            "agent": "job_matching",
            "is_re_evaluation": False,
            "classification": bias_classification,
            "specific_feedback": specific_feedback
        }
        
        return {
            "candidate_id": candidate_id,
            "dataset_index": candidate_data.get('dataset_index', 0),
            "role": candidate_data.get('Role', job_requirements.get('title', 'Unknown Role')),
            "final_decision": final_decision,
            "bias_classification": bias_classification,
            "re_evaluation_count": 0,
            "evaluation_insights": [evaluation_insight],
            "processing_time": time.strftime('%Y-%m-%dT%H:%M:%S.%f', time.gmtime()),
            "workflow_completed": True,
            "job_feedback_count": 1,
            "bias_feedback_count": 1,
            "ground_truth_decision": candidate_data.get('ground_truth_decision', final_decision),
            "ground_truth_bias": candidate_data.get('ground_truth_bias', bias_classification),
            # Legacy fields for compatibility
            "job_match": job_analysis,
            "bias_analysis": bias_analysis,
            "processing_time_seconds": {
                "job_analysis_seconds": job_time,
                "bias_analysis_seconds": bias_time,
                "total_seconds": total_time
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/batch_analyze")
async def batch_analyze(request: Dict[str, Any]):
    """Analyze multiple candidates in parallel (smaller batches)"""
    
    if not job_agent or not bias_agent:
        raise HTTPException(status_code=503, detail="Agents not initialized")
    
    try:
        candidates = request.get('candidates', [])
        job_requirements = request.get('job_requirements', {})
        
        if not candidates:
            raise HTTPException(status_code=400, detail="No candidates provided")
        
        if len(candidates) > 50:  # Limit batch size for API
            raise HTTPException(status_code=400, detail="Batch size too large (max 50 candidates)")
        
        start_time = time.time()
        
        # Process candidates with limited concurrency
        semaphore = asyncio.Semaphore(Config.CONCURRENT_REQUESTS)
        
        async def process_single(candidate):
            async with semaphore:
                return await analyze_single_candidate_async(candidate, job_requirements)
        
        tasks = [process_single(candidate) for candidate in candidates]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        successful_results = []
        failed_count = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process candidate {i}: {result}")
                failed_count += 1
                successful_results.append({
                    "candidate_id": candidates[i].get('id', f'candidate_{i}'),
                    "error": str(result),
                    "job_match": None,
                    "bias_analysis": None
                })
            else:
                successful_results.append(result)
        
        total_time = time.time() - start_time
        
        return {
            "total_candidates": len(candidates),
            "successful_analyses": len(successful_results) - failed_count,
            "failed_analyses": failed_count,
            "processing_time_seconds": total_time,
            "average_time_per_candidate": total_time / len(candidates),
            "results": successful_results,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

async def analyze_single_candidate_async(candidate_data: Dict[str, Any], job_requirements: Dict[str, Any]):
    """Async wrapper for candidate analysis"""
    loop = asyncio.get_event_loop()
    
    try:
        # Run in thread pool to avoid blocking
        job_analysis = await loop.run_in_executor(
            None, 
            job_agent.run,
            candidate_data['Resume'],
            candidate_data['Job_Description'], 
            candidate_data['Transcript'],
            candidate_data['Role']
        )
        
        bias_analysis = await loop.run_in_executor(
            None,
            bias_agent.run,
            candidate_data['Resume'],
            candidate_data['Job_Description'],
            candidate_data['Transcript'],
            job_analysis.get('decision', 'unknown'),
            candidate_data['Role'],
            job_analysis.get('primary_reason', '')
        )
        
        return {
            "candidate_id": candidate_data.get('id', 'unknown'),
            "job_match": job_analysis,
            "bias_analysis": bias_analysis,
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise Exception(f"Failed to analyze candidate: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """Get system performance metrics"""
    try:
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Process metrics
        process = psutil.Process()
        process_memory = process.memory_info()
        
        metrics = {
            "timestamp": time.time(),
            "uptime_seconds": time.time() - startup_time if startup_time else 0,
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3)
            },
            "process": {
                "memory_rss_mb": process_memory.rss / (1024**2),
                "memory_vms_mb": process_memory.vms / (1024**2),
                "cpu_percent": process.cpu_percent()
            }
        }
        
        return metrics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

if __name__ == "__main__":
    # Set environment variables for RunPod
    os.environ.setdefault('OLLAMA_BASE_URL', 'http://localhost:11434')
    os.environ.setdefault('MODEL_NAME', 'gemma3:27b-instruct')
    os.environ.setdefault('RUNPOD_POD_ID', 'local-development')
    os.environ.setdefault('WORKSPACE_PATH', os.getcwd())
    
    # RunPod-optimized server configuration
    uvicorn.run(
        "runpod_main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload in production
        workers=1,     # Single worker for GPU efficiency
        log_level=Config.LOG_LEVEL.lower(),
        access_log=True
    )
