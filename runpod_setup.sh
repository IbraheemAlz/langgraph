#!/bin/bash
"""
RunPod Environment Setup Script
==============================
Sets up the complete environment for the AI Hiring System on RunPod
"""

echo "🚀 Setting up AI Hiring System on RunPod..."
echo "📅 $(date)"

# Navigate to workspace
cd /workspace || exit 1

# Clone repository if not exists
if [ ! -d "langgraph" ]; then
    echo "📥 Cloning repository..."
    git clone https://github.com/IbraheemAlz/langgraph.git || {
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

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt || {
    echo "❌ Failed to install Python dependencies"
    exit 1
}

# Set up environment variables
echo "⚙️ Setting up environment..."
cat > .env << EOF
# RunPod Environment Configuration
OLLAMA_BASE_URL=http://localhost:11434
MODEL_NAME=gemma3:27b-instruct
RUNPOD_POD_ID=${RUNPOD_POD_ID:-unknown}
WORKSPACE_PATH=/workspace/langgraph
LOG_LEVEL=INFO
MAX_WORKERS=4
BATCH_SIZE=3
CONCURRENT_REQUESTS=2
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

echo ""
echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Run: python run_on_runpod.py"
echo "2. Wait for model download (10-15 minutes for first run)"
echo "3. Access your app at http://[pod-ip]:8000"
echo ""
echo "📊 For batch processing:"
echo "  • Use: python runpod_batch_processor.py --input your_data.csv"
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
