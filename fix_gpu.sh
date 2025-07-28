#!/bin/bash
# GPU Fix for Ollama on RunPod H100
# Based on RunPod documentation recommendations

echo "🎯 Fixing Ollama GPU Configuration for H100..."

# Step 1: Install lshw as recommended by RunPod docs
echo "🔧 Installing lshw for GPU hardware detection..."
apt update -qq
apt install -y lshw

# Step 2: Stop existing Ollama processes
echo "🛑 Stopping existing Ollama processes..."
pkill -f ollama || true
sleep 3
pkill -9 -f ollama || true
sleep 2

# Step 3: Set GPU environment variables
echo "⚙️ Setting H100 GPU environment variables..."
export CUDA_VISIBLE_DEVICES=0
export OLLAMA_GPU_OVERHEAD=0
export OLLAMA_NUM_GPU=1
export OLLAMA_MAX_LOADED_MODELS=1

# Step 4: Check GPU status
echo "🔍 Checking GPU status..."
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

# Step 5: Start Ollama with proper GPU detection
echo "🚀 Starting Ollama with GPU optimization..."
nohup /usr/local/bin/ollama serve > ollama_gpu.log 2>&1 &
OLLAMA_PID=$!
echo "Started Ollama with PID: $OLLAMA_PID"

# Step 6: Wait for Ollama to start
echo "⏳ Waiting for Ollama to start..."
for i in {1..30}; do
    if curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
        echo "✅ Ollama is ready"
        break
    fi
    sleep 1
done

# Step 7: Force model reload with GPU
echo "🔄 Forcing model reload with full GPU utilization..."
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma3:27b",
    "prompt": "Force GPU reload",
    "options": {
      "num_gpu": 99,
      "num_thread": 1
    },
    "stream": false,
    "keep_alive": 0
  }' > /dev/null 2>&1

# Wait and try again with keep alive
sleep 5

curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma3:27b", 
    "prompt": "Test GPU",
    "options": {
      "num_gpu": 99,
      "num_thread": 1
    },
    "stream": false
  }' > /dev/null 2>&1 &

# Step 8: Check GPU utilization
sleep 10
echo "🎯 Checking GPU utilization during inference..."
nvidia-smi

echo "📋 Ollama logs (last 20 lines):"
tail -20 ollama_gpu.log

echo ""
echo "🎉 GPU fix complete!"
echo "🔧 To stop Ollama: pkill -f ollama"
echo "📊 Monitor GPU: watch nvidia-smi"
echo "📝 Ollama logs: tail -f ollama_gpu.log"
