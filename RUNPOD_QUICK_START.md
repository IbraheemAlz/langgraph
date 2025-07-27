# RunPod Quick Start Guide

## 🚀 Simple Steps to Deploy Your AI Hiring System on RunPod

### Step 1: Create RunPod Pod

1. **Go to RunPod.io** and log in
2. **Click "Pods" in the side menu**
3. **Select GPU**: Choose **A100 PCIe** (40GB VRAM)
4. **GPU Count**: Keep at **1**
5. **Template**: Keep **"Runpod PyTorch 2.8.0"**

### Step 2: Configure Pricing
- **For Testing**: Select **"On-Demand"** ($1.64/hr)
- **For Production**: Select **"Spot"** ($0.82/hr - 50% cheaper)

### Step 3: Enable Required Options
- ✅ **Keep "Start Jupyter Notebook" checked**
- ⚠️ **Leave "SSH Terminal Access" unchecked** (we'll use Jupyter only)

### Step 4: Edit Template Settings
**Click "Edit Template" and configure:**

- **Container Disk**: Change to **50 GB**
- **Volume Disk**: Change to **100 GB** 
- **HTTP Ports**: Enter `8888,11434,8000`
- **Volume Mount Path**: Keep `/workspace`
- **Environment Variables**: Add these:
  ```
  OLLAMA_HOST=0.0.0.0
  OLLAMA_PORT=11434
  JUPYTER_ENABLE_LAB=yes
  ```

**Click "Set Overrides"**

### Step 5: Deploy Pod
- **Click "Deploy"**
- **Wait 2-3 minutes** for pod to start

---

## 💻 Setup Your System (After Pod Starts)

### Step 1: Connect to Your Pod
1. **Click "Connect"** on your running pod
2. **Choose "Connect to Jupyter Lab"** 
3. **Open a Terminal in Jupyter** (New → Terminal)

### Step 2: Install Ollama Directly
**The PyTorch template already includes GPU support! Just install Ollama directly:**
```bash
# Install Ollama directly (no Docker needed!)
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve &

# Wait a moment for service to start
sleep 5

# Download Gemma 3 model (this takes 10-15 minutes)
ollama pull gemma3:27b-instruct

# Verify model is ready
ollama list
```

### Step 3: Deploy Your Application
**In the Jupyter Terminal:**
```bash
# Navigate to workspace
cd /workspace

# Clone your repository
git clone https://github.com/IbraheemAlz/langgraph.git
cd langgraph

# Install Python dependencies
pip install -r requirements.txt

# Create simple run script
cat > run_on_runpod.py << 'EOF'
import os
import sys
sys.path.append('/workspace/langgraph/src')

from main import app
import uvicorn

if __name__ == "__main__":
    # Configure for RunPod
    os.environ['OLLAMA_BASE_URL'] = 'http://localhost:11434'
    os.environ['MODEL_NAME'] = 'gemma3:27b-instruct'
    
    print("🚀 Starting AI Hiring System on RunPod...")
    print("📊 Access at: http://[your-pod-ip]:8000")
    print("📓 Jupyter at: http://[your-pod-ip]:8888")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

# Make executable
chmod +x run_on_runpod.py
```

### Step 4: Update Configuration for RunPod
**Create the updated config file in Jupyter:**
```bash
# Update config file for local Ollama
cat > src/config.py << 'EOF'
import os

class Config:
    # Use local Ollama instead of API
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    MODEL_NAME = os.getenv('MODEL_NAME', 'gemma3:27b-instruct')
    
    # Remove API key requirement for local deployment
    USE_LOCAL_MODEL = True
    GEMINI_API_KEY = None  # Not needed for local Ollama
    
    # Optimization settings for A100
    MAX_WORKERS = 4
    BATCH_SIZE = 3  # Process 3 candidates simultaneously
    
    # Performance tuning
    MODEL_CONTEXT_LENGTH = 8192
    TEMPERATURE = 0.1
    TOP_P = 0.9
    
    # File paths
    DATA_FOLDER = "data"
    RESULTS_FOLDER = "results"
EOF
```

---

## ▶️ Run Your System

### Start the Application
**In the Jupyter Terminal:**
```bash
cd /workspace/langgraph
python run_on_runpod.py
```

### Test Your System
**Open another terminal in Jupyter (New → Terminal) and run:**
```bash
# In another terminal, test batch processing
python batch_processor.py --input sample-data.csv
```

---

## 🔗 Access Your System

**After deployment, you can access:**
- **AI Hiring System**: `http://[your-pod-ip]:8000`
- **Jupyter Notebook**: `http://[your-pod-ip]:8888`
- **Ollama API**: `http://[your-pod-ip]:11434`

**Find your Pod IP:**
- In RunPod dashboard, click on your pod
- Copy the "TCP Port Mapping" IP address

---

## 📊 Expected Performance

| Metric | Expected Value |
|--------|----------------|
| **Processing Speed** | 400-600 candidates/hour |
| **Cost per Hour** | $0.82 (Spot) / $1.64 (On-Demand) |
| **Startup Time** | 15-20 minutes (including model download) |
| **GPU Memory Usage** | ~35GB / 40GB available |

---

## 🛠️ Troubleshooting

### If Ollama won't start:
```bash
pkill ollama
ollama serve &
sleep 5
```

### If model download fails:
```bash
ollama pull gemma3:27b-instruct
```

### Check GPU usage:
```bash
nvidia-smi
```

### If application won't start:
```bash
cd /workspace/langgraph
pip install --upgrade -r requirements.txt
python run_on_runpod.py
```

---

## 💡 Tips for Success

1. **First Time Setup**: Allow 20-30 minutes for complete setup
2. **Model Download**: Gemma 3 27B takes ~15 minutes to download
3. **Cost Optimization**: Use "Spot" instances for 50% savings
4. **Storage**: 100GB volume ensures you have space for models and data
5. **Performance**: A100 gives you the best speed for this workload
6. **Jupyter Interface**: All commands run through Jupyter Lab terminal - no SSH needed

---

## 🎯 Quick Commands Summary

**All commands run in Jupyter Lab Terminal (New → Terminal):**

```bash
# Essential commands after pod starts (no Docker needed!):
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
sleep 5
ollama pull gemma3:27b-instruct
cd /workspace && git clone https://github.com/IbraheemAlz/langgraph.git
cd langgraph && pip install -r requirements.txt
python run_on_runpod.py
```

**🚀 That's it! Your AI Hiring System is now running on RunPod with local Gemma 3 model for maximum performance and cost efficiency.**
