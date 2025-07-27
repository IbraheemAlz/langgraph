#!/usr/bin/env python3
"""
RunPod deployment entry point for AI Hiring System
Handles Ollama setup, model download, and application startup
"""

import os
import sys
import time
import logging
import subprocess
import signal
import requests
from pathlib import Path

def setup_logging():
    """Setup logging for RunPod deployment"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('/workspace/langgraph/runpod_startup.log')
        ]
    )
    return logging.getLogger(__name__)

def check_ollama_service():
    """Check if Ollama service is running"""
    logger = logging.getLogger(__name__)
    
    try:
        response = requests.get('http://localhost:11434/api/version', timeout=5)
        if response.status_code == 200:
            logger.info("✅ Ollama service is running")
            return True
    except:
        pass
    
    logger.warning("⚠️ Ollama service not detected")
    return False

def start_ollama():
    """Start Ollama service if not running"""
    logger = logging.getLogger(__name__)
    
    if check_ollama_service():
        return True
    
    logger.info("🚀 Starting Ollama service...")
    
    try:
        # Start Ollama in background
        env = dict(os.environ)
        env.update({
            "OLLAMA_HOST": "0.0.0.0",
            "OLLAMA_PORT": "11434",
            "OLLAMA_ORIGINS": "*"
        })
        
        process = subprocess.Popen(
            ['ollama', 'serve'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env
        )
        
        # Wait for service to start
        for i in range(30):
            time.sleep(2)
            if check_ollama_service():
                logger.info("✅ Ollama service started successfully")
                return True
            logger.info(f"⏳ Waiting for Ollama to start... ({i+1}/30)")
        
        logger.error("❌ Failed to start Ollama service")
        return False
        
    except Exception as e:
        logger.error(f"❌ Error starting Ollama: {e}")
        return False

def check_model():
    """Check if Gemma model is available"""
    logger = logging.getLogger(__name__)
    
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=10)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [model.get('name', '') for model in models]
            
            if any('gemma3:27b-instruct' in name for name in model_names):
                logger.info("✅ Gemma 3 model is available")
                return True
            else:
                logger.warning(f"⚠️ Gemma 3 model not found. Available models: {model_names}")
                return False
    except Exception as e:
        logger.error(f"❌ Error checking model: {e}")
        return False

def download_model():
    """Download Gemma 3 model if not available"""
    logger = logging.getLogger(__name__)
    
    if check_model():
        return True
    
    logger.info("📥 Downloading Gemma 3 model...")
    logger.info("⏰ This may take 10-15 minutes for the 27B model")
    
    try:
        # Download model with progress monitoring
        process = subprocess.Popen(
            ['ollama', 'pull', 'gemma3:27b-instruct'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Monitor progress
        last_progress_time = time.time()
        for line in process.stdout:
            line = line.strip()
            if line:
                # Log progress every 30 seconds to avoid spam
                current_time = time.time()
                if current_time - last_progress_time > 30:
                    logger.info(f"📥 Download progress: {line}")
                    last_progress_time = current_time
        
        process.wait()
        
        if process.returncode == 0:
            logger.info("✅ Model downloaded successfully")
            return check_model()  # Verify it's actually available
        else:
            logger.error(f"❌ Model download failed with exit code: {process.returncode}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error downloading model: {e}")
        return False

def install_dependencies():
    """Install Python dependencies"""
    logger = logging.getLogger(__name__)
    
    logger.info("📦 Installing Python dependencies...")
    
    try:
        # Install from requirements.txt
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info("✅ Dependencies installed successfully")
            return True
        else:
            logger.error(f"❌ Dependency installation failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error installing dependencies: {e}")
        return False

def start_application():
    """Start the FastAPI application"""
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting AI Hiring System application...")
    
    # Set environment variables
    os.environ.update({
        'OLLAMA_BASE_URL': 'http://localhost:11434',
        'MODEL_NAME': 'gemma3:27b-instruct',
        'RUNPOD_POD_ID': os.getenv('RUNPOD_POD_ID', 'unknown'),
        'WORKSPACE_PATH': '/workspace/langgraph',
        'LOG_LEVEL': 'INFO'
    })
    
    try:
        # Change to the correct directory
        os.chdir('/workspace/langgraph')
        
        # Start the application
        logger.info("✅ Application starting on http://0.0.0.0:8000")
        logger.info("📊 Access points:")
        logger.info("   • Main API: http://[pod-ip]:8000")
        logger.info("   • Health Check: http://[pod-ip]:8000/health")
        logger.info("   • API Docs: http://[pod-ip]:8000/docs")
        logger.info("   • Metrics: http://[pod-ip]:8000/metrics")
        
        # Run the application
        subprocess.run([
            sys.executable, '-m', 'uvicorn', 'runpod_main:app',
            '--host', '0.0.0.0',
            '--port', '8000',
            '--log-level', 'info'
        ])
        
    except KeyboardInterrupt:
        logger.info("🛑 Application stopped by user")
    except Exception as e:
        logger.error(f"❌ Failed to start application: {e}")
        return False

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger = logging.getLogger(__name__)
    logger.info("🛑 Shutdown signal received")
    sys.exit(0)

def main():
    """Main entry point for RunPod deployment"""
    # Setup
    logger = setup_logging()
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("🎯 Starting AI Hiring System on RunPod")
    logger.info(f"📍 Pod ID: {os.getenv('RUNPOD_POD_ID', 'unknown')}")
    logger.info(f"💾 Workspace: /workspace/langgraph")
    
    # Pre-flight checks
    logger.info("🔍 Running pre-flight checks...")
    
    # Check workspace
    if not Path('/workspace/langgraph').exists():
        logger.error("❌ Workspace not found at /workspace/langgraph")
        logger.info("💡 Please ensure the repository is cloned to /workspace/langgraph")
        return 1
    
    # Change to workspace directory
    os.chdir('/workspace/langgraph')
    
    # Install dependencies
    if not install_dependencies():
        logger.error("❌ Failed to install dependencies")
        return 1
    
    # Start Ollama
    if not start_ollama():
        logger.error("❌ Failed to start Ollama service")
        return 1
    
    # Download model
    if not download_model():
        logger.error("❌ Failed to download model")
        logger.info("💡 You can try downloading manually: ollama pull gemma3:27b-instruct")
        return 1
    
    # Final health check
    logger.info("🏥 Final health check...")
    if not check_ollama_service() or not check_model():
        logger.error("❌ System not ready")
        return 1
    
    logger.info("🎉 All systems ready! Starting application...")
    
    # Start application
    start_application()
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
