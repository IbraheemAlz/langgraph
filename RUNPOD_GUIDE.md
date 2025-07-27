# RunPod AI Hiring System - Complete Guide

## 🚀 Quick Setup

### 1. Create RunPod Pod

- **GPU**: H100 PCIe (94GB VRAM)
- **Template**: PyTorch 2.8.0
- **Storage**: 100GB volume
- **Ports**: 8000, 8888, 11434

### 2. Setup Environment

```bash
cd /workspace
git clone -b runpod https://github.com/IbraheemAlz/langgraph.git
cd langgraph
chmod +x runpod_setup.sh
./runpod_setup.sh
```

### 3. Launch Application

```bash
python run_on_runpod.py
```

### 4. Process Your Data

```bash
python runpod_batch_processor.py --input your_data.csv
```

---

## 📁 File Structure

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
- **Memory Usage**: 90% of 94GB VRAM
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

---

## 🎯 API Endpoints

Once running, access at `http://[pod-ip]:8000`:

- **API Documentation**: `/docs`
- **Health Check**: `/health`
- **Metrics**: `/metrics`
- **Process Single Candidate**: `POST /process`
- **Batch Processing Status**: `GET /status`

---

## 💰 Performance & Cost

### H100 PCIe Performance

- **Processing**: 1,200-1,800 candidates/hour
- **Total Time**: 6-8.5 hours for 10,175 candidates
- **Cost**: ~$10-14 total (spot pricing)
- **Speedup**: 3-4X faster than A100

### Expected Results

Your AI system will complete 10K+ candidate evaluations in under 8 hours with comprehensive bias detection and job matching analysis.

---

**🎉 Your H100-optimized AI Hiring System is ready for production use!**
