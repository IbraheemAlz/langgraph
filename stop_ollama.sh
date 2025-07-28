#!/bin/bash
# Stop Ollama processes

echo "🛑 Stopping all Ollama processes..."

# Get all ollama process IDs
PIDS=$(pgrep -f ollama)

if [ -z "$PIDS" ]; then
    echo "ℹ️ No Ollama processes found running"
else
    echo "📋 Found Ollama processes: $PIDS"
    
    # Graceful shutdown first
    echo "🔄 Attempting graceful shutdown..."
    pkill -TERM -f ollama
    sleep 5
    
    # Force kill if still running
    REMAINING=$(pgrep -f ollama)
    if [ ! -z "$REMAINING" ]; then
        echo "⚡ Force killing remaining processes: $REMAINING"
        pkill -KILL -f ollama
        sleep 2
    fi
    
    # Final check
    FINAL_CHECK=$(pgrep -f ollama)
    if [ -z "$FINAL_CHECK" ]; then
        echo "✅ All Ollama processes stopped successfully"
    else
        echo "❌ Some processes may still be running: $FINAL_CHECK"
    fi
fi

echo "🔍 Current GPU status:"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits
