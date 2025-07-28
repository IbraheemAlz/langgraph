#!/bin/bash
# Quick System Verification Script

echo "🔍 System Verification Report"
echo "============================"

# Check GPU
echo "🎯 GPU Status:"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | head -1

# Check Ollama
echo ""
echo "🤖 Ollama Status:"
if curl -s http://localhost:11434/api/version > /dev/null; then
    echo "✅ Ollama service: RUNNING"
    
    # Check models
    MODELS=$(curl -s http://localhost:11434/api/tags | grep -o '"name":"[^"]*"' | cut -d'"' -f4)
    if [[ "$MODELS" == *"gemma3:27b"* ]]; then
        echo "✅ Gemma 3 27B model: AVAILABLE"
    else
        echo "❌ Gemma 3 27B model: NOT FOUND"
    fi
    
    # Check GPU layer utilization from logs
    if [ -f "ollama.log" ] || [ -f "ollama_gpu.log" ]; then
        LAYERS=$(grep "offloaded.*layers to GPU" ollama*.log 2>/dev/null | tail -1)
        if [ ! -z "$LAYERS" ]; then
            echo "🎯 GPU Layers: $LAYERS"
        fi
    fi
else
    echo "❌ Ollama service: NOT RUNNING"
fi

# Check Application
echo ""
echo "🚀 Application Status:"
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ AI Hiring System: RUNNING"
    
    HEALTH=$(curl -s http://localhost:8000/health)
    if [[ "$HEALTH" == *"healthy"* ]]; then
        echo "✅ Health Status: HEALTHY"
    else
        echo "⚠️ Health Status: UNKNOWN"
    fi
else
    echo "❌ AI Hiring System: NOT RUNNING"
fi

# Check processes
echo ""
echo "🔄 Running Processes:"
echo "Ollama: $(pgrep -f ollama | wc -l) processes"
echo "Python: $(pgrep -f python | wc -l) processes"

# Check ports
echo ""
echo "🔗 Port Status:"
echo "Port 11434 (Ollama): $(netstat -ln | grep :11434 | wc -l) listeners"
echo "Port 8000 (API): $(netstat -ln | grep :8000 | wc -l) listeners"

echo ""
echo "📊 Quick Performance Test:"
echo "Testing model response time..."
START_TIME=$(date +%s.%N)
RESPONSE=$(curl -s -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma3:27b",
    "prompt": "Hello",
    "options": {"num_predict": 5},
    "stream": false
  }' 2>/dev/null)
END_TIME=$(date +%s.%N)

if [ ! -z "$RESPONSE" ] && [[ "$RESPONSE" != *"error"* ]]; then
    DURATION=$(echo "$END_TIME - $START_TIME" | bc)
    echo "✅ Response time: ${DURATION}s"
    
    if (( $(echo "$DURATION < 10" | bc -l) )); then
        echo "🚀 Performance: EXCELLENT"
    elif (( $(echo "$DURATION < 20" | bc -l) )); then
        echo "⚡ Performance: GOOD"
    else
        echo "🐌 Performance: SLOW (may need GPU fix)"
    fi
else
    echo "❌ Model test: FAILED"
fi

echo ""
echo "🎯 Overall Status:"
if curl -s http://localhost:8000/health > /dev/null && curl -s http://localhost:11434/api/version > /dev/null; then
    echo "🎉 System is READY for production!"
else
    echo "❌ System needs attention"
fi
