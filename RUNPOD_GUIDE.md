# RunPod Deployment Guide for AI Hiring System

## Quick Start

### 1. Start the System
```bash
python run_on_runpod.py
```

### 2. Run Batch Processing
```bash
python runpod_batch_processor.py --input sample-data.csv
```

### 3. Monitor Performance (NEW!)
```bash
python performance_monitor.py
```

## 🔄 **SMART RESUME FUNCTIONALITY (NEW!)**

The batch processor now automatically **resumes from where it left off** if processing is interrupted!

### How Resume Works:
1. **Automatic Detection**: Scans for existing JSON files in `results/json/` folder
2. **ID Matching**: Reads `candidate_id` from existing results
3. **Smart Filtering**: Skips already processed candidates automatically
4. **Seamless Continuation**: Continues processing remaining candidates

### Resume Examples:
```bash
# First run (processes 1000 candidates, gets interrupted after 400)
python runpod_batch_processor.py --input data.csv

# Second run (automatically skips first 400, processes remaining 600)  
python runpod_batch_processor.py --input data.csv

# Force reprocess ALL candidates (ignore existing results)
python runpod_batch_processor.py --input data.csv --force
```

### What You'll See:
```
📂 Found existing output file: results/json/runpod_batch_results_1738027800.json
✅ Already processed: 847 candidates
🔄 Filtering results:
   📊 Total candidates: 10174
   ✅ Already processed: 847
   🔄 Remaining to process: 9327
🔄 RESUMING processing from 847 existing results
📊 Remaining: 9327 candidates to process
```

## Recent Fixes (2025-07-28)

### ✅ Fixed Ollama Configuration Warnings
- Removed invalid Ollama options: `max_tokens`, `gpu_memory_utilization`, `use_mlock`  
- Updated to use correct Ollama parameters:
  - `num_predict` instead of `max_tokens`
  - Removed `gpu_memory_utilization` (not supported by Ollama)
  - Removed `use_mlock` (deprecated option)
- Valid options now used: `temperature`, `top_p`, `num_ctx`, `num_batch`, `num_gpu`, `num_thread`, `use_mmap`, `numa`, `repeat_penalty`, `top_k`, `num_predict`

### ✅ Fixed Batch Processor JSON Output Issue
- Added robust directory creation with proper permissions
- Implemented multiple fallback strategies for file saving:
  1. Primary location: `results/json/`
  2. Temporary directory: `/tmp/`
  3. Current directory: `./`
  4. Workspace directory: `/workspace/`
- Added file verification after saving
- Enhanced error handling and logging
- Console output as final backup if all file saves fail

### 🔧 Enhanced RunPod Environment Detection
- Automatic detection of RunPod vs local environment
- Better logging for debugging
- Workspace path validation
- Pod ID tracking

## Usage Commands

### Basic Commands (H100-Optimized)
```bash
# Start the H100-optimized AI system (required first)
python run_on_runpod.py

# Run batch processing with H100 optimizations
python runpod_batch_processor.py --input sample-data.csv

# Monitor performance in real-time
python performance_monitor.py

# Resume interrupted processing (automatic)
python runpod_batch_processor.py --input sample-data.csv --output existing_results.json

# Force reprocess all candidates (ignore existing results)
python runpod_batch_processor.py --input sample-data.csv --force

# Run with custom job requirements
python runpod_batch_processor.py \
  --input sample-data.csv \
  --job-title "Senior Developer" \
  --required-skills "Python,React,AWS,Docker" \
  --experience-level "Senior" \
  --education "Bachelor's degree required"

# Specify custom output location
python runpod_batch_processor.py \
  --input sample-data.csv \
  --output /workspace/my_results.json
```

### Health Check
```bash
curl http://localhost:8000/health
```

## Expected Output

After running the batch processor, you should see:
1. **Environment Detection**: Pod ID and workspace path
2. **API Health Check**: Confirmation that the service is running  
3. **Processing Progress**: Batch-by-batch progress with rates
4. **Multiple Save Attempts**: The system will try different locations to save results
5. **Final Summary**: Success rate, total time, and file location

Example successful output:
```
🏃 Running in environment: RunPod
📍 Pod ID: abc123xyz
📁 Workspace: /workspace/langgraph
✅ Batch processor initialized successfully
🔍 Checking API health...
✅ API service healthy
📂 Input file: sample-data.csv
📊 Loaded 1812 candidates
⚡ Processing with 10 concurrent requests
📦 Batch size: 20 candidates per batch
📊 Processing batch 1/91 (20 candidates)
✅ Batch 1 complete in 15.2s
📈 Progress: 20/1812 (1.3 candidates/sec)
...
💾 Attempting to save results to primary output location: results/json/runpod_batch_results_1738027800.json
✅ Results successfully saved to: results/json/runpod_batch_results_1738027800.json
🎉 Batch processing complete!
📊 Results Summary:
  • Total candidates: 1812
  • Successful: 1810
  • Failed: 2
  • Success rate: 99.9%
  • Total time: 1247.3s
  • Average rate: 1.5 candidates/sec
📊 Result file size: 2,847,392 bytes
✅ Processing complete!
```

## Troubleshooting

### If Ollama warnings appear:
The recent fixes should eliminate these warnings. If you still see warnings about invalid options, ensure you're using the updated agent files.

### If JSON output fails:
The system now tries multiple locations automatically:
1. Check `/workspace/` directory
2. Check `/tmp/` directory  
3. Check current directory for `runpod_batch_results_*.json`
4. Look for console output as backup

### If API is not healthy:
```bash
# Restart the system
python run_on_runpod.py
```

### Common Issues:
- **Permission denied**: The system now automatically handles permissions
- **Directory not found**: Multiple fallback directories are created automatically
- **API not responding**: Check that `run_on_runpod.py` completed successfully

## File Structure

Your results will be saved in JSON format with this structure:
```json
{
  "metadata": {
    "total_candidates": 1812,
    "successful_analyses": 1810,
    "processing_time_seconds": 1247.3,
    "model_used": "gemma3:27b",
    "pod_id": "abc123xyz"
  },
  "results": [
    {
      "candidate_id": "lisaro937",
      "final_decision": "select",
      "bias_classification": "unbiased",
      "evaluation_insights": [...],
      "workflow_completed": true
    }
  ],
  "summary": {
    "success_rate": 99.9,
    "avg_processing_time_per_candidate": 0.69
  }
}
```
