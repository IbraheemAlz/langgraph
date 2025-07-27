# RunPod AI Hiring System - Complete Guide

## 🚀 Quick Setup

### Step 1: Create RunPod Pod

1. **Go to RunPod.io** and log in
2. **Click "Pods" in the side menu**
3. **Select GPU**: Choose **H100 PCIe** (80GB VRAM)
4. **GPU Count**: Keep at **1**
5. **Template**: Keep **"Runpod PyTorch 2.8.0"**

### Step 2: Configure Pricing

- **For Testing**: Select **"On-Demand"** (~$2.39/hr)
- **For Production**: Select **"Spot"** (~$2.03/hr - 15% cheaper)

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

### Step 2: Automated Setup (Recommended)

**In the Jupyter Terminal:**

```bash
cd /workspace
git clone -b runpod https://github.com/IbraheemAlz/langgraph.git
cd langgraph
chmod +x runpod_setup.sh
./runpod_setup.sh
```

### Step 3: Launch Application

```bash
python run_on_runpod.py
```

### Step 4: Process Your Data

```bash
python runpod_batch_processor.py --input your_data.csv
```

---

## � Access Your System

**After deployment, you can access:**

- **AI Hiring System**: `http://[your-pod-ip]:8000`
- **API Documentation**: `http://[your-pod-ip]:8000/docs`
- **Health Check**: `http://[your-pod-ip]:8000/health`
- **Metrics**: `http://[your-pod-ip]:8000/metrics`
- **Jupyter Notebook**: `http://[your-pod-ip]:8888`
- **Ollama API**: `http://[your-pod-ip]:11434`

**Find your Pod IP:**

- In RunPod dashboard, click on your pod
- Copy the "TCP Port Mapping" IP address

---

## 📊 Expected Performance

| Metric               | H100 PCIe Performance                    |
| -------------------- | ---------------------------------------- |
| **Processing Speed** | 1,200-1,800 candidates/hour              |
| **Cost per Hour**    | $2.03 (Spot) / $2.39 (On-Demand)         |
| **Startup Time**     | 10-15 minutes (including model download) |
| **GPU Memory Usage** | ~50GB / 80GB available                   |
| **Total Time (10K)** | 6-8 hours                                |

---

## �📁 File Structure

```
/workspace/langgraph/
├── src/                          # Core AI system
│   ├── config.py                 # H100-optimized configuration
│   ├── main.py                   # LangGraph workflow
│   └── agents/                   # AI agents
├── runpod_setup.sh              # Complete setup script
├── run_on_runpod.py             # Application launcher
├── runpod_main.py               # FastAPI server
├── runpod_batch_processor.py    # Main processing engine
├── test_ollama_setup.py         # System verification
├── merged_data_final.csv        # Your dataset
└── results/                     # Processing results
    └── json/                    # JSON output files
```

---

## 🔧 Configuration

### H100 Performance Settings (Pre-configured)

- **Processing Speed**: 1,200-1,800 candidates/hour
- **Parallel Workers**: 12
- **Batch Size**: 10 candidates simultaneously
- **Memory Usage**: 90% of 80GB VRAM
- **Expected Time**: 6-8 hours for 10K candidates

### Environment Variables (Auto-configured)

```bash
MAX_WORKERS=12
BATCH_SIZE=10
CONCURRENT_REQUESTS=6
MODEL_CONTEXT_LENGTH=4096
TEMPERATURE=0.01
MAX_TOKENS=512
REQUEST_TIMEOUT=30
```

---

## 📊 Usage Commands

### System Verification

```bash
python test_ollama_setup.py
```

### Monitor Processing

```bash
tail -f results/batch_processing.log
```

### Check GPU Usage

```bash
nvidia-smi
```

### Check Ollama Models

```bash
ollama list
```

---

## 🛠️ Troubleshooting

### If Setup Fails

```bash
# Check logs
cat runpod_startup.log

# Manual Ollama install
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull gemma3:27b-instruct
```

### If Ollama won't start

```bash
pkill ollama
ollama serve &
sleep 5
```

### If Processing Fails

```bash
# Verify system
python test_ollama_setup.py

# Check agent imports
python -c "from src.agents.job_matching_agent import JobMatchingAgent; print('✅ Agents working')"
```

### If Model Download Fails

```bash
# Check available space
df -h

# Retry download
ollama pull gemma3:27b-instruct
```

### Check GPU Usage

```bash
nvidia-smi
```

---

## 💡 Tips for Success

1. **First Time Setup**: Allow 15-20 minutes for complete setup
2. **Model Download**: Gemma 3 27B takes ~10 minutes on H100
3. **Cost Optimization**: Use "Spot" instances for 50% savings
4. **Storage**: 100GB volume ensures space for models and data
5. **Performance**: H100 gives maximum speed for this workload
6. **Jupyter Interface**: All commands run through Jupyter Lab terminal

---

## 🎯 Quick Commands Summary

**Essential commands after pod starts (run in Jupyter Terminal):**

```bash
cd /workspace
git clone -b runpod https://github.com/IbraheemAlz/langgraph.git
cd langgraph
chmod +x runpod_setup.sh
./runpod_setup.sh
python run_on_runpod.py
```

**🚀 That's it! Your H100-optimized AI Hiring System is now running for maximum performance!**

---

## ✅ Success Indicators

### Setup Complete

- `ollama list` shows `gemma3:27b-instruct`
- `python test_ollama_setup.py` shows all green checkmarks
- `nvidia-smi` shows model loaded (~50GB memory used)

### Processing Working

- Speed: 1,200-1,800 candidates/hour in logs
- Regular updates in `results/batch_processing.log`
- JSON files appearing in `results/json/`
- GPU: 80-90% utilization
