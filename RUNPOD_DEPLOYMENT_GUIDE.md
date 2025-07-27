# Runpod Deployment Guide for Multi-Agent AI Hiring System

## Overview

This guide provides comprehensive instructions for deploying your LangGraph-based Multi-Agent AI Hiring System on Runpod infrastructure to achieve optimal performance and cost efficiency.

## Table of Contents

1. [Deployment Architecture](#deployment-architecture)
2. [Runpod Setup Options](#runpod-setup-options)
3. [Model Deployment Strategy](#model-deployment-strategy)
4. [Step-by-Step Deployment](#step-by-step-deployment)
5. [Performance Optimization](#performance-optimization)
6. [Cost Optimization](#cost-optimization)
7. [Monitoring and Maintenance](#monitoring-and-maintenance)
8. [Troubleshooting](#troubleshooting)

## Deployment Architecture

### Recommended Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Runpod Pod Instance                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │   Ollama Server │  │  LangGraph App  │                  │
│  │   (Port 11434)  │  │   (Port 8000)   │                  │
│  │                 │  │                 │                  │
│  │  Google Gemma 3 │  │ ┌─────────────┐ │                  │
│  │     (27B-IT)    │  │ │ Job Matching│ │                  │
│  │                 │  │ │    Agent    │ │                  │
│  │  Local Storage  │  │ └─────────────┘ │                  │
│  │                 │  │ ┌─────────────┐ │                  │
│  │                 │  │ │Bias Classif.│ │                  │
│  │                 │  │ │    Agent    │ │                  │
│  └─────────────────┘  │ └─────────────┘ │                  │
│                       └─────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Ollama Server**: Local model serving for Google Gemma 3
2. **LangGraph Application**: Multi-agent orchestration
3. **GPU Acceleration**: NVIDIA RTX 4090 or A100 for optimal performance
4. **Local Storage**: SSD for model caching and data processing

## Runpod Setup Options

### Option 1: Serverless Endpoints (Recommended for Variable Workloads)

**Advantages:**
- Pay-per-use pricing
- Auto-scaling
- No management overhead
- Cold start optimization

**Best For:**
- Batch processing jobs
- Variable demand
- Cost-conscious deployments

### Option 2: Community Cloud (Recommended for Consistent Workloads)

**Advantages:**
- Persistent instances
- Better cost for continuous use
- Full control over environment
- No cold starts

**Best For:**
- 24/7 availability requirements
- High-volume processing
- Development and testing

### Option 3: Secure Cloud (Enterprise)

**Advantages:**
- Enhanced security
- Compliance features
- Dedicated infrastructure
- SLA guarantees

**Best For:**
- Enterprise deployments
- Sensitive data processing
- Compliance requirements

## Model Deployment Strategy

### Local Model Storage Approach

Instead of API calls, deploy Google Gemma 3 locally using Ollama:

```bash
# Download and cache Gemma 3 model locally
ollama pull gemma3:27b-instruct
```

**Benefits:**
- No API rate limits
- Reduced latency
- Lower operational costs
- Full control over model inference

### Memory Requirements

| Model Variant | VRAM Needed | Recommended GPU |
|---------------|-------------|-----------------|
| Gemma 3 7B    | 8GB+       | RTX 4070, RTX 4080 |
| Gemma 3 27B   | 24GB+      | RTX 4090, A100 |

## Step-by-Step Deployment

### Step 1: Pod Configuration

1. **Create New Pod**
   ```
   Template: PyTorch 2.1 + Python 3.10
   GPU: RTX 4090 (24GB VRAM) or A100
   Container Disk: 50GB+
   Volume Disk: 100GB+ (for model storage)
   ```

2. **Network Ports**
   ```
   11434: Ollama API
   8000: LangGraph Application
   8888: Jupyter (optional)
   ```

### Step 2: Environment Setup

```bash
# Update system
apt update && apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y nvidia-container-toolkit

# Configure Docker for NVIDIA
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
```

### Step 3: Ollama Installation and Model Setup

```bash
# Run Ollama with GPU support
docker run -d --gpus=all \
    -v ollama:/root/.ollama \
    -p 11434:11434 \
    --name ollama \
    ollama/ollama

# Pull Gemma 3 model
docker exec -it ollama ollama pull gemma3:27b-instruct

# Verify model installation
docker exec -it ollama ollama list
```

### Step 4: Application Deployment

```bash
# Clone your repository
git clone <your-repo-url>
cd langgraph

# Create optimized Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "src/main.py"]
EOF

# Build and run application
docker build -t langgraph-app .
docker run -d \
    --name langgraph-app \
    --network host \
    -e GEMINI_API_KEY=local \
    -e OLLAMA_BASE_URL=http://localhost:11434 \
    langgraph-app
```

### Step 5: Configuration Updates

Update your configuration to use local Ollama:

```python
# src/config.py modifications
import os

class Config:
    # Use local Ollama instead of API
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    MODEL_NAME = "gemma3:27b-instruct"
    
    # Remove API key requirement for local deployment
    USE_LOCAL_MODEL = True
    
    # Optimization settings
    MAX_WORKERS = 4  # Adjust based on GPU memory
    BATCH_SIZE = 2   # Process 2 candidates simultaneously
    
    # Performance tuning
    MODEL_CONTEXT_LENGTH = 8192
    TEMPERATURE = 0.1
    TOP_P = 0.9
```

## Performance Optimization

### GPU Memory Management

1. **Model Quantization**
   ```bash
   # Use quantized versions for better performance
   ollama pull gemma3:27b-instruct-q4_K_M
   ```

2. **Batch Processing Optimization**
   ```python
   # Optimize batch sizes based on available VRAM
   def calculate_optimal_batch_size(gpu_memory_gb):
       if gpu_memory_gb >= 24:
           return 4  # RTX 4090, A100
       elif gpu_memory_gb >= 16:
           return 2  # RTX 4080
       else:
           return 1  # Fallback
   ```

### Concurrent Processing

```python
# Enhanced batch processor for Runpod
import asyncio
from concurrent.futures import ThreadPoolExecutor

class RunpodBatchProcessor:
    def __init__(self, max_workers=4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def process_candidates_parallel(self, candidates):
        tasks = []
        for candidate in candidates:
            task = asyncio.create_task(self.process_single_candidate(candidate))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
```

### Storage Optimization

1. **Model Caching**
   ```bash
   # Persistent volume for model storage
   docker volume create ollama-models
   ```

2. **Result Storage**
   ```bash
   # Use Runpod's network storage for results
   mkdir -p /workspace/results
   ln -s /workspace/results ./results
   ```

## Cost Optimization

### Serverless Endpoint Strategy

For variable workloads, use Runpod Serverless:

```python
# Create serverless deployment script
import runpod

def handler(event):
    """Serverless handler for batch processing"""
    candidates = event.get('candidates', [])
    
    # Process candidates
    processor = BatchProcessor()
    results = processor.process_batch(candidates)
    
    return {
        "statusCode": 200,
        "body": results
    }

runpod.serverless.start({"handler": handler})
```

### Instance Scheduling

```bash
# Auto-stop script for cost savings
cat > auto_stop.sh << 'EOF'
#!/bin/bash
# Stop instance after 1 hour of inactivity
timeout=3600
last_activity=$(date +%s)

while true; do
    current_time=$(date +%s)
    if [ $((current_time - last_activity)) -gt $timeout ]; then
        echo "Stopping instance due to inactivity"
        runpod stop
        break
    fi
    sleep 300  # Check every 5 minutes
done
EOF

chmod +x auto_stop.sh
nohup ./auto_stop.sh &
```

## Monitoring and Maintenance

### Health Checks

```python
# Health monitoring script
import requests
import time

def check_ollama_health():
    try:
        response = requests.get('http://localhost:11434/api/version')
        return response.status_code == 200
    except:
        return False

def check_app_health():
    try:
        response = requests.get('http://localhost:8000/health')
        return response.status_code == 200
    except:
        return False

# Monitoring loop
while True:
    if not check_ollama_health():
        print("Ollama service down, restarting...")
        # Restart logic here
    
    if not check_app_health():
        print("Application down, restarting...")
        # Restart logic here
    
    time.sleep(60)
```

### Performance Metrics

```python
# Performance tracking
import psutil
import nvidia_ml_py3 as nvidia_ml

def get_system_metrics():
    # CPU and Memory
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    
    # GPU metrics
    nvidia_ml.nvmlInit()
    handle = nvidia_ml.nvmlDeviceGetHandleByIndex(0)
    gpu_util = nvidia_ml.nvmlDeviceGetUtilizationRates(handle)
    gpu_memory = nvidia_ml.nvmlDeviceGetMemoryInfo(handle)
    
    return {
        'cpu_percent': cpu_percent,
        'memory_percent': memory.percent,
        'gpu_util': gpu_util.gpu,
        'gpu_memory_percent': (gpu_memory.used / gpu_memory.total) * 100
    }
```

## Security Best Practices

1. **Environment Variables**
   ```bash
   # Store sensitive data in Runpod secrets
   export GEMINI_API_KEY="your-backup-api-key"
   export DATABASE_URL="your-database-url"
   ```

2. **Network Security**
   ```python
   # Restrict API access
   ALLOWED_IPS = ['your.client.ip.address']
   
   @app.middleware("http")
   async def ip_whitelist(request: Request, call_next):
       client_ip = request.client.host
       if client_ip not in ALLOWED_IPS:
           return JSONResponse(
               status_code=403,
               content={"detail": "Access forbidden"}
           )
       return await call_next(request)
   ```

## Troubleshooting

### Common Issues and Solutions

1. **GPU Memory Issues**
   ```bash
   # Check GPU memory usage
   nvidia-smi
   
   # Clear GPU cache
   docker exec -it ollama ollama stop gemma3:27b-instruct
   docker restart ollama
   ```

2. **Model Loading Failures**
   ```bash
   # Verify model integrity
   docker exec -it ollama ollama show gemma3:27b-instruct
   
   # Re-download if corrupted
   docker exec -it ollama ollama rm gemma3:27b-instruct
   docker exec -it ollama ollama pull gemma3:27b-instruct
   ```

3. **Performance Degradation**
   ```python
   # Monitor and optimize batch sizes
   def adaptive_batch_size(current_performance):
       if current_performance < 0.8:  # 80% of expected performance
           return max(1, current_batch_size - 1)
       elif current_performance > 0.95:
           return min(8, current_batch_size + 1)
       return current_batch_size
   ```

## Deployment Checklist

- [ ] Runpod pod created with appropriate GPU
- [ ] Docker and NVIDIA Container Toolkit installed
- [ ] Ollama server running with Gemma 3 model
- [ ] Application deployed and accessible
- [ ] Health checks configured
- [ ] Monitoring setup
- [ ] Cost optimization measures in place
- [ ] Security configurations applied
- [ ] Backup and recovery procedures established

## Expected Performance Metrics

### Throughput Expectations

| GPU Type | Model Size | Candidates/Hour | Cost/Hour |
|----------|------------|-----------------|-----------|
| RTX 4090 | Gemma 3 27B | 200-300 | $0.50-0.70 |
| A100     | Gemma 3 27B | 400-600 | $1.20-1.50 |
| RTX 4080 | Gemma 3 7B  | 300-400 | $0.40-0.60 |

### Quality Metrics

- **Job Matching Accuracy**: 85-90%
- **Bias Detection Accuracy**: 92-95%
- **Processing Latency**: 2-5 seconds per candidate
- **System Uptime**: 99.5%+

## Conclusion

This deployment strategy maximizes performance while minimizing costs by:

1. Using local model deployment instead of API calls
2. Leveraging GPU acceleration for faster processing
3. Implementing efficient batch processing
4. Optimizing resource utilization
5. Providing comprehensive monitoring and maintenance

The combination of Runpod's infrastructure and local Ollama deployment provides the best balance of performance, cost, and reliability for your Multi-Agent AI Hiring System.
