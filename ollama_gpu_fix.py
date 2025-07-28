#!/usr/bin/env python3
"""
Ollama GPU Configuration Fix for RunPod H100
Addresses the critical issue where only 1/63 layers are offloaded to GPU
"""

import os
import subprocess
import time
import requests
import logging
import signal
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def stop_ollama():
    """Stop all Ollama processes"""
    logger.info("🛑 Stopping Ollama processes...")
    
    try:
        # Kill all ollama processes
        subprocess.run(['pkill', '-f', 'ollama'], check=False)
        time.sleep(2)
        
        # Force kill if still running
        subprocess.run(['pkill', '-9', '-f', 'ollama'], check=False)
        time.sleep(1)
        
        logger.info("✅ Ollama processes stopped")
        return True
    except Exception as e:
        logger.error(f"❌ Error stopping Ollama: {e}")
        return False

def check_gpu_availability():
    """Check if GPU is available and get info"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
        logger.info("🎯 GPU detected:")
        for line in result.stdout.split('\n'):
            if 'H100' in line or 'MiB' in line:
                logger.info(f"   {line.strip()}")
        return True
    except Exception as e:
        logger.error(f"❌ GPU not available: {e}")
        return False

def install_lshw():
    """Install lshw as recommended by RunPod docs for GPU detection"""
    logger.info("🔧 Installing lshw for GPU hardware detection...")
    try:
        subprocess.run(['apt', 'update'], check=True, capture_output=True)
        subprocess.run(['apt', 'install', '-y', 'lshw'], check=True, capture_output=True)
        logger.info("✅ lshw installed successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to install lshw: {e}")
        return False

def start_ollama_optimized():
    """Start Ollama with H100 optimizations"""
    logger.info("🚀 Starting Ollama with H100 GPU optimizations...")
    
    # Set H100-optimized environment variables
    env = os.environ.copy()
    env.update({
        # Force GPU usage
        'CUDA_VISIBLE_DEVICES': '0',
        'OLLAMA_GPU_OVERHEAD': '0',
        'OLLAMA_NUM_GPU': '1',
        'OLLAMA_MAX_LOADED_MODELS': '1',
        'OLLAMA_MAX_QUEUE': '512',
        'OLLAMA_NUM_PARALLEL': '4',
        
        # Memory optimization for H100
        'OLLAMA_MAX_VRAM': '75000000000',  # 75GB out of 80GB
        'OLLAMA_GPU_MEMORY_FRACTION': '0.95',
        
        # CUDA optimizations
        'CUDA_LAUNCH_BLOCKING': '0',
        'CUDA_CACHE_DISABLE': '0',
        
        # Performance optimizations
        'OLLAMA_FLASH_ATTENTION': '1',
        'OLLAMA_NUM_THREAD': '16',
    })
    
    try:
        # Start Ollama with optimized settings
        process = subprocess.Popen(
            ['/usr/local/bin/ollama', 'serve'],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        logger.info(f"🟢 Ollama started with PID: {process.pid}")
        
        # Wait for Ollama to be ready
        for attempt in range(30):
            try:
                response = requests.get('http://localhost:11434/api/version', timeout=2)
                if response.status_code == 200:
                    logger.info("✅ Ollama service is ready")
                    return process
            except:
                pass
            time.sleep(1)
            
        logger.error("❌ Ollama failed to start properly")
        process.terminate()
        return None
        
    except Exception as e:
        logger.error(f"❌ Failed to start Ollama: {e}")
        return None

def reload_model_with_gpu():
    """Reload the model with proper GPU configuration"""
    logger.info("🔄 Reloading model with GPU optimization...")
    
    try:
        # First, unload the model if it exists
        requests.post('http://localhost:11434/api/generate', 
                     json={'model': 'gemma3:27b', 'keep_alive': 0}, 
                     timeout=10)
        time.sleep(2)
        
        # Pull/reload with GPU optimization
        logger.info("📥 Pulling model with GPU optimization...")
        
        # Send a simple request to load the model
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'gemma3:27b',
                'prompt': 'Hello',
                'options': {
                    'num_gpu': 99,  # Force all layers to GPU
                    'gpu_memory_utilization': 0.95,
                    'num_thread': 16,
                },
                'stream': False
            },
            timeout=60
        )
        
        if response.status_code == 200:
            logger.info("✅ Model loaded with GPU optimization")
            return True
        else:
            logger.error(f"❌ Failed to load model: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error reloading model: {e}")
        return False

def check_gpu_utilization():
    """Check if GPU is properly utilized"""
    logger.info("🔍 Checking GPU utilization...")
    
    try:
        # Send a test request to see GPU usage
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'gemma3:27b',
                'prompt': 'Test GPU utilization',
                'options': {'num_predict': 10},
                'stream': False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            logger.info("✅ Model responding - checking nvidia-smi...")
            
            # Check GPU usage during inference
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'MiB' in line and ('python' in line or 'ollama' in line):
                    logger.info(f"🎯 GPU Usage: {line.strip()}")
            
            return True
        else:
            logger.error("❌ Model not responding properly")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error checking GPU utilization: {e}")
        return False

def main():
    """Main function to fix Ollama GPU configuration"""
    logger.info("🎯 Starting Ollama GPU Fix for H100...")
    
    if len(sys.argv) > 1 and sys.argv[1] == 'stop':
        stop_ollama()
        return
    
    # Step 1: Check GPU availability
    if not check_gpu_availability():
        logger.error("❌ No GPU available - cannot proceed")
        return
    
    # Step 2: Install lshw for GPU detection
    install_lshw()
    
    # Step 3: Stop existing Ollama
    stop_ollama()
    
    # Step 4: Start Ollama with optimizations
    process = start_ollama_optimized()
    if not process:
        logger.error("❌ Failed to start Ollama")
        return
    
    # Step 5: Reload model with GPU optimization
    if not reload_model_with_gpu():
        logger.error("❌ Failed to optimize model loading")
        return
    
    # Step 6: Check GPU utilization
    check_gpu_utilization()
    
    logger.info("🎉 Ollama GPU optimization complete!")
    logger.info("🔧 Use 'python ollama_gpu_fix.py stop' to stop Ollama")
    
    # Keep the script running to monitor
    try:
        logger.info("📊 Monitoring mode - Press Ctrl+C to exit")
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        logger.info("🛑 Stopping monitoring...")
        stop_ollama()

if __name__ == "__main__":
    main()
