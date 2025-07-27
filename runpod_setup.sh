#!/bin/bash
#
# RunPod Environment Setup Script
# ==============================
# Sets up the complete environment for the AI Hiring System on RunPod
#

echo "🚀 Setting up AI Hiring System on RunPod..."
echo "📅 $(date)"

# Navigate to workspace
cd /workspace || exit 1

# Clone repository if not exists
if [ ! -d "langgraph" ]; then
    echo "📥 Cloning repository..."
    git clone -b runpod https://github.com/IbraheemAlz/langgraph.git || {
        echo "❌ Failed to clone repository"
        exit 1
    }
else
    echo "✅ Repository already exists"
fi

cd langgraph || exit 1

# Switch to runpod branch if it exists
if git branch -r | grep -q "origin/runpod"; then
    echo "🔄 Switching to runpod branch..."
    git checkout runpod
    git pull origin runpod
else
    echo "ℹ️ Using main branch"
fi

# Create results directory structure
echo "📁 Creating directories..."
mkdir -p results/json
mkdir -p logs

# Install system dependencies
echo "🔧 Installing system dependencies..."
apt-get update -qq
apt-get install -y curl wget git

# Install Ollama
echo "🤖 Installing Ollama..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
    echo "✅ Ollama installed"
else
    echo "✅ Ollama already installed"
fi

# Start Ollama service
echo "🚀 Starting Ollama service..."
pkill ollama 2>/dev/null || true
ollama serve &
sleep 10

# Verify Ollama is running
echo "🔍 Verifying Ollama service..."
if curl -s http://localhost:11434/api/version >/dev/null 2>&1; then
    echo "✅ Ollama service is running"
else
    echo "❌ Ollama service failed to start"
    exit 1
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt || {
    echo "❌ Failed to install Python dependencies"
    exit 1
}

# Set up environment variables for H100 optimization
echo "⚙️ Setting up H100-optimized environment..."
cat > .env << EOF
# H100 PCIe Optimized Configuration
OLLAMA_BASE_URL=http://localhost:11434
MODEL_NAME=gemma3:27b
RUNPOD_POD_ID=${RUNPOD_POD_ID:-unknown}
WORKSPACE_PATH=/workspace/langgraph
LOG_LEVEL=INFO

# H100 PERFORMANCE SETTINGS
MAX_WORKERS=12
BATCH_SIZE=10
CONCURRENT_REQUESTS=6

# H100 OLLAMA OPTIMIZATIONS
OLLAMA_NUM_GPU=1
OLLAMA_NUM_THREAD=16
OLLAMA_BATCH_SIZE=1024
OLLAMA_FLASH_ATTENTION=1
OLLAMA_GPU_MEMORY_FRACTION=0.9

# H100 MODEL PARAMETERS
MODEL_CONTEXT_LENGTH=4096
TEMPERATURE=0.01
TOP_P=0.7
MAX_TOKENS=512
REQUEST_TIMEOUT=30
EOF

# Make scripts executable
chmod +x run_on_runpod.py
chmod +x runpod_batch_processor.py
chmod +x runpod_setup.sh

# Test Python imports
echo "🧪 Testing Python imports..."
python3 -c "
import sys
sys.path.append('src')
try:
    from config import Config
    print('✅ Config import successful')
except Exception as e:
    print(f'❌ Config import failed: {e}')
    exit(1)
"

# Download the AI model
echo "📥 Downloading Gemma 3 model..."
echo "⏰ This may take 5-10 minutes on H100..."
if ollama pull gemma3:27b; then
    echo "✅ Model downloaded successfully"
else
    echo "❌ Model download failed"
    echo "💡 You can try downloading manually: ollama pull gemma3:27b"
fi

# Verify model is available
echo "🔍 Verifying model availability..."
if ollama list | grep -q "gemma3:27b"; then
    echo "✅ Gemma 3 model is ready"
else
    echo "⚠️ Model verification failed - check manually with: ollama list"
fi

echo ""
echo "🎉 H100 Setup Complete!"
echo ""
echo "🎯 Your system is ready! Next steps:"
echo "1. Run: python run_on_runpod.py"
echo "2. Access your app at http://[pod-ip]:8000"
echo "3. For batch processing: python runpod_batch_processor.py --input your_data.csv"
echo ""
echo "📊 Expected Performance:"
echo "  • Processing: 1,200-1,800 candidates/hour"
echo "  • Time for 10K candidates: 6-8 hours"
echo "  • GPU Usage: 80-90% on H100"
echo ""
echo "🔧 Useful commands:"
echo "  • Check Ollama: ollama list"
echo "  • Check GPU: nvidia-smi" 
echo "  • View logs: tail -f runpod_startup.log"
echo ""
echo "📖 Documentation:"
echo "  • API docs: http://[pod-ip]:8000/docs"
echo "  • Health check: http://[pod-ip]:8000/health"
echo "  • Metrics: http://[pod-ip]:8000/metrics"
