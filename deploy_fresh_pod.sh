#!/bin/bash
# Complete Fresh Pod Deployment Script
# Combines setup, GPU optimization, and application startup

set -e  # Exit on any error

echo "🎯 Starting Fresh Pod Deployment for AI Hiring System"
echo "=================================================="

# Step 1: Pull latest changes
echo "📥 Pulling latest changes from repository..."
git pull origin runpod || echo "⚠️ Git pull failed or no changes"

# Step 2: Make scripts executable
echo "🔧 Making scripts executable..."
chmod +x runpod_setup.sh fix_gpu.sh stop_ollama.sh

# Step 3: Run initial setup
echo "⚙️ Running initial setup..."
./runpod_setup.sh

# Step 4: Check if GPU fix is needed
echo "🔍 Checking GPU utilization..."
sleep 10

# Test if model is using GPU properly
echo "🧪 Testing model GPU usage..."
RESPONSE=$(curl -s -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma3:27b",
    "prompt": "Test",
    "options": {"num_predict": 5},
    "stream": false
  }' || echo "FAILED")

if [[ "$RESPONSE" == *"FAILED"* ]] || [ -z "$RESPONSE" ]; then
    echo "❌ Model not responding properly - applying GPU fix..."
    ./fix_gpu.sh
else
    echo "✅ Model responding - checking GPU layer utilization..."
    
    # Check logs for GPU layer usage
    if grep -q "offloaded 1/63 layers to GPU" ollama.log 2>/dev/null; then
        echo "❌ Only 1/63 layers on GPU - applying GPU fix..."
        ./fix_gpu.sh
    else
        echo "✅ GPU utilization looks good"
    fi
fi

# Step 5: Final health check
echo "🏥 Final system health check..."
sleep 5

# Check Ollama
if curl -s http://localhost:11434/api/version > /dev/null; then
    echo "✅ Ollama service is running"
else
    echo "❌ Ollama service is not responding"
    exit 1
fi

# Check model availability
if curl -s http://localhost:11434/api/tags | grep -q "gemma3:27b"; then
    echo "✅ Gemma 3 27B model is available"
else
    echo "❌ Gemma 3 27B model is not available"
    exit 1
fi

# Step 6: Start the application
echo "🚀 Starting AI Hiring System application..."
python run_on_runpod.py &
APP_PID=$!

# Wait for application to start
echo "⏳ Waiting for application startup..."
sleep 15

# Final verification
if curl -s http://localhost:8000/health > /dev/null; then
    echo ""
    echo "🎉 DEPLOYMENT SUCCESSFUL!"
    echo "================================"
    echo "🔗 Application URL: http://[pod-ip]:8000"
    echo "📊 Health Check: http://[pod-ip]:8000/health"
    echo "📚 API Docs: http://[pod-ip]:8000/docs"
    echo "🔧 Application PID: $APP_PID"
    echo ""
    echo "🎯 Ready for batch processing:"
    echo "   python runpod_batch_processor.py --input sample-data.csv"
    echo ""
    echo "🛠️ Management commands:"
    echo "   ./stop_ollama.sh  # Stop Ollama"
    echo "   ./fix_gpu.sh      # Fix GPU utilization"
    echo "   kill $APP_PID     # Stop application"
    echo ""
    echo "📈 Monitor GPU: watch nvidia-smi"
else
    echo "❌ Application failed to start properly"
    echo "🔍 Check logs: tail -f runpod_deployment.log"
    exit 1
fi
