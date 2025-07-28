#!/usr/bin/env python3
"""
RunPod Optimized Batch Processor for Multi-Agent AI Hiring System
Designed for high-throughput processing of large datasets (10K+ candidates)
"""

import asyncio
import aiohttp
import json
import time
import logging
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
import sys
import argparse
import os
import stat

# Add src to path
sys.path.append('src')
from src.config import Config

class RunPodBatchProcessor:
    """Optimized batch processor for RunPod deployment"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.config = Config()
        self.logger = self._setup_logging()
        self.concurrent_limit = self.config.CONCURRENT_REQUESTS
        
        # Initialize output directories and verify permissions
        self._setup_output_directories()
        
    def _setup_logging(self):
        """Setup logging for batch processing"""
        logging.basicConfig(
            level=getattr(logging, self.config.LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f'{self.config.RESULTS_FOLDER}/batch_processing.log')
            ]
        )
        return logging.getLogger(__name__)
        
    def _setup_output_directories(self):
        """Setup and verify output directories with proper permissions"""
        directories_to_create = [
            self.config.RESULTS_FOLDER,
            f"{self.config.RESULTS_FOLDER}/json",
            "/tmp",  # Fallback directory
            "/workspace"  # RunPod workspace directory
        ]
        
        for dir_path in directories_to_create:
            try:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                # Test write permissions
                test_file = Path(dir_path) / "test_write_permission.tmp"
                test_file.write_text("test")
                test_file.unlink()  # Remove test file
                self.logger.info(f"✅ Directory ready with write access: {dir_path}")
            except Exception as e:
                self.logger.warning(f"⚠️ Directory access issue for {dir_path}: {e}")
                continue
        
    async def process_batch_file(self, input_file: str, job_requirements: Dict[str, Any], output_file: str = None):
        """Process a CSV file of candidates with optimized batching"""
        
        self.logger.info(f"🚀 Starting RunPod batch processing")
        self.logger.info(f"📂 Input file: {input_file}")
        
        # Read and validate input file
        try:
            df = pd.read_csv(input_file)
            candidates = df.to_dict('records')
            
            # Enrich candidate data with dataset_index and proper ID mapping
            for i, candidate in enumerate(candidates):
                if 'dataset_index' not in candidate:
                    candidate['dataset_index'] = i
                
                # Handle ID mapping - CSV has 'ID' (uppercase), API expects 'id' (lowercase)
                if 'ID' in candidate and candidate['ID']:
                    candidate['id'] = candidate['ID']  # Map CSV 'ID' to API 'id'
                    self.logger.debug(f"Mapped ID: {candidate['ID']} → {candidate['id']}")
                elif 'id' not in candidate:
                    candidate['id'] = f"candidate_{i}"  # Fallback for missing ID
                    self.logger.debug(f"Generated fallback ID: {candidate['id']}")
            
            self.logger.info(f"📊 Loaded {len(candidates)} candidates")
            self.logger.info(f"🔍 Sample IDs: {[c.get('id', 'missing') for c in candidates[:3]]}")  # Show first 3 IDs
        except Exception as e:
            self.logger.error(f"❌ Failed to load input file: {e}")
            return None
        
        if len(candidates) == 0:
            self.logger.error("❌ No candidates found in input file")
            return None
        
        # Setup processing
        start_time = time.time()
        batch_size = self.config.BATCH_SIZE
        all_results = []
        
        self.logger.info(f"⚡ Processing with {self.concurrent_limit} concurrent requests")
        self.logger.info(f"📦 Batch size: {batch_size} candidates per batch")
        
        # Process in optimized batches
        total_batches = (len(candidates) + batch_size - 1) // batch_size
        
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            self.logger.info(f"📊 Processing batch {batch_num}/{total_batches} ({len(batch)} candidates)")
            batch_start = time.time()
            
            try:
                batch_results = await self._process_batch_async(batch, job_requirements)
                all_results.extend(batch_results)
                
                # Progress metrics
                batch_time = time.time() - batch_start
                processed = len(all_results)
                total_elapsed = time.time() - start_time
                rate = processed / total_elapsed if total_elapsed > 0 else 0
                eta = (len(candidates) - processed) / rate if rate > 0 else 0
                
                self.logger.info(f"✅ Batch {batch_num} complete in {batch_time:.1f}s")
                self.logger.info(f"📈 Progress: {processed}/{len(candidates)} ({rate:.1f} candidates/sec)")
                self.logger.info(f"⏱️ ETA: {eta:.0f}s remaining")
                
                # Memory management for large datasets
                if batch_num % 10 == 0:
                    self.logger.info("🧹 Running memory cleanup...")
                    import gc
                    gc.collect()
                
            except Exception as e:
                self.logger.error(f"❌ Batch {batch_num} failed: {e}")
                # Add failed results with error info
                for candidate in batch:
                    all_results.append({
                        "candidate_id": candidate.get('id', f'batch_{batch_num}_candidate'),
                        "error": str(e),
                        "job_match": None,
                        "bias_analysis": None,
                        "timestamp": time.time()
                    })
        
        # Generate output file
        if output_file is None:
            timestamp = int(time.time())
            output_file = f"{self.config.RESULTS_FOLDER}/json/runpod_batch_results_{timestamp}.json"
        
        # Ensure output directory exists with proper permissions
        output_dir = Path(output_file).parent
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            # Ensure we have write permissions
            os.chmod(output_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            self.logger.info(f"📁 Output directory prepared: {output_dir}")
        except Exception as e:
            self.logger.error(f"❌ Failed to create output directory {output_dir}: {e}")
            # Fallback to current directory
            output_file = f"runpod_batch_results_{timestamp}.json"
            self.logger.info(f"📁 Using fallback output file: {output_file}")
        
        # Calculate final metrics
        total_time = time.time() - start_time
        successful_results = [r for r in all_results if r.get('error') is None]
        failed_results = [r for r in all_results if r.get('error') is not None]
        
        # Filter and clean results for expected format
        cleaned_results = []
        for result in successful_results:
            if result.get('error') is None:
                # Create clean result matching expected format
                cleaned_result = {
                    "candidate_id": result.get('candidate_id', 'unknown'),
                    "dataset_index": result.get('dataset_index', 0),
                    "role": result.get('role', 'Unknown Role'),
                    "final_decision": result.get('final_decision', 'reject'),
                    "bias_classification": result.get('bias_classification', 'unbiased'),
                    "re_evaluation_count": result.get('re_evaluation_count', 0),
                    "evaluation_insights": result.get('evaluation_insights', []),
                    "processing_time": result.get('processing_time', ''),
                    "workflow_completed": result.get('workflow_completed', True),
                    "job_feedback_count": result.get('job_feedback_count', 1),
                    "bias_feedback_count": result.get('bias_feedback_count', 1),
                    "ground_truth_decision": result.get('ground_truth_decision', 'unknown'),
                    "ground_truth_bias": result.get('ground_truth_bias', 'unknown')
                }
                cleaned_results.append(cleaned_result)
        
        # Save comprehensive results with better error handling
        output_data = {
            "metadata": {
                "total_candidates": len(candidates),
                "successful_analyses": len(successful_results),
                "failed_analyses": len(failed_results),
                "processing_time_seconds": total_time,
                "average_rate_per_second": len(candidates) / total_time,
                "job_requirements": job_requirements,
                "model_used": self.config.MODEL_NAME,
                "pod_id": self.config.RUNPOD_POD_ID,
                "batch_size": batch_size,
                "concurrent_requests": self.concurrent_limit,
                "timestamp": time.time()
            },
            "results": cleaned_results,
            "summary": {
                "total_processed": len(cleaned_results),
                "success_rate": len(cleaned_results) / len(all_results) * 100 if all_results else 0,
                "avg_processing_time_per_candidate": total_time / len(candidates) if candidates else 0
            }
        }
        
        # Try to save the results with multiple fallback strategies
        save_success = False
        attempts = [
            (output_file, "primary output location"),
            (f"/tmp/runpod_batch_results_{timestamp}.json", "temporary directory"),
            (f"./runpod_batch_results_{timestamp}.json", "current directory"),
            (f"/workspace/runpod_batch_results_{timestamp}.json", "workspace directory")
        ]
        
        for attempt_file, attempt_desc in attempts:
            try:
                self.logger.info(f"💾 Attempting to save results to {attempt_desc}: {attempt_file}")
                
                # Create directory if needed
                attempt_dir = Path(attempt_file).parent
                attempt_dir.mkdir(parents=True, exist_ok=True)
                
                # Write the file
                with open(attempt_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                
                # Verify file was created and has content
                if Path(attempt_file).exists() and Path(attempt_file).stat().st_size > 0:
                    output_file = attempt_file  # Update the final output file path
                    save_success = True
                    self.logger.info(f"✅ Results successfully saved to: {output_file}")
                    break
                else:
                    self.logger.warning(f"⚠️ File created but appears empty: {attempt_file}")
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to save to {attempt_desc}: {e}")
                continue
        
        if not save_success:
            self.logger.error("❌ Failed to save results to any location!")
            # As a last resort, print the results to console
            self.logger.info("📄 Printing results to console as backup:")
            print(json.dumps(output_data, indent=2))
            output_file = None
        
        # Final summary
        self.logger.info("🎉 Batch processing complete!")
        self.logger.info(f"📊 Results Summary:")
        self.logger.info(f"  • Total candidates: {len(candidates)}")
        self.logger.info(f"  • Successful: {len(cleaned_results)}")
        self.logger.info(f"  • Failed: {len(failed_results)}")
        self.logger.info(f"  • Success rate: {len(cleaned_results)/len(candidates)*100:.1f}%")
        self.logger.info(f"  • Total time: {total_time:.1f}s")
        self.logger.info(f"  • Average rate: {len(candidates)/total_time:.1f} candidates/sec")
        self.logger.info(f"💾 Results saved to: {output_file}")
        
        return output_file
    
    async def _process_batch_async(self, candidates: List[Dict[str, Any]], job_requirements: Dict[str, Any]):
        """Process a batch of candidates asynchronously with optimized concurrency"""
        
        # Use semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.concurrent_limit)
        
        async def process_single(candidate):
            async with semaphore:
                return await self._analyze_candidate_api(candidate, job_requirements)
        
        # Create tasks for all candidates in batch
        tasks = [process_single(candidate) for candidate in candidates]
        
        # Process with timeout and exception handling
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.config.REQUEST_TIMEOUT * len(candidates)
            )
        except asyncio.TimeoutError:
            self.logger.error("Batch processing timeout")
            results = [Exception("Batch timeout")] * len(candidates)
        
        # Handle exceptions and format results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to process candidate {i}: {result}")
                processed_results.append({
                    "candidate_id": candidates[i].get('id', f'candidate_{i}'),
                    "error": str(result),
                    "job_match": None,
                    "bias_analysis": None,
                    "timestamp": time.time()
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _analyze_candidate_api(self, candidate: Dict[str, Any], job_requirements: Dict[str, Any]):
        """Make API call to analyze single candidate with retry logic"""
        
        # Debug: Log candidate ID being sent to API
        candidate_id = candidate.get('id', 'missing')
        self.logger.debug(f"🔄 API call for candidate ID: {candidate_id}")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                timeout = aiohttp.ClientTimeout(total=self.config.REQUEST_TIMEOUT)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        f"{self.api_url}/analyze_candidate",
                        json={
                            "candidate_data": candidate,
                            "job_requirements": job_requirements
                        }
                    ) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            error_text = await response.text()
                            raise Exception(f"API error {response.status}: {error_text}")
                            
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Timeout on attempt {attempt + 1}, retrying...")
                    await asyncio.sleep(1)
                    continue
                else:
                    raise Exception("Request timeout after retries")
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Error on attempt {attempt + 1}: {e}, retrying...")
                    await asyncio.sleep(1)
                    continue
                else:
                    raise Exception(f"API call failed after retries: {e}")
    
    async def health_check(self):
        """Check if the API service is healthy"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/health", timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        health_data = await response.json()
                        self.logger.info(f"✅ API service healthy: {health_data}")
                        return True
                    else:
                        self.logger.error(f"❌ API service unhealthy: {response.status}")
                        return False
        except Exception as e:
            self.logger.error(f"❌ Health check failed: {e}")
            return False

def main():
    """CLI interface for RunPod batch processing"""
    parser = argparse.ArgumentParser(description="RunPod Batch Processor for AI Hiring System")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", help="Output JSON file path (optional)")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--job-title", default="Software Engineer", help="Job title")
    parser.add_argument("--required-skills", default="Python,JavaScript,AI,Machine Learning", help="Required skills (comma-separated)")
    parser.add_argument("--experience-level", default="Mid-level", help="Experience level")
    parser.add_argument("--education", default="Bachelor's degree preferred", help="Education requirements")
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input).exists():
        print(f"❌ Input file not found: {args.input}")
        return 1
    
    # Check if we're in RunPod environment
    runpod_pod_id = os.environ.get('RUNPOD_POD_ID', 'local')
    workspace_path = os.environ.get('WORKSPACE_PATH', os.getcwd())
    print(f"🏃 Running in environment: {'RunPod' if runpod_pod_id != 'local' else 'Local'}")
    print(f"📍 Pod ID: {runpod_pod_id}")
    print(f"📁 Workspace: {workspace_path}")
    
    # Create job requirements
    job_requirements = {
        "title": args.job_title,
        "required_skills": [skill.strip() for skill in args.required_skills.split(",")],
        "experience_level": args.experience_level,
        "education_requirements": args.education
    }
    
    # Initialize processor
    try:
        processor = RunPodBatchProcessor(api_url=args.api_url)
        print(f"✅ Batch processor initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize batch processor: {e}")
        return 1
    
    async def run_processing():
        # Health check first
        print("🔍 Checking API health...")
        if not await processor.health_check():
            print("❌ API service not healthy. Please ensure the service is running.")
            print("💡 Try running: python run_on_runpod.py")
            return 1
        
        # Run batch processing
        try:
            result_file = await processor.process_batch_file(
                args.input,
                job_requirements,
                args.output
            )
            
            if result_file:
                print(f"✅ Processing complete! Results saved to: {result_file}")
                # Verify file exists and has content
                if Path(result_file).exists():
                    file_size = Path(result_file).stat().st_size
                    print(f"📊 Result file size: {file_size:,} bytes")
                    return 0
                else:
                    print(f"⚠️ Result file was not found at expected location: {result_file}")
                    return 1
            else:
                print("❌ Processing failed - no output file created")
                return 1
                
        except Exception as e:
            print(f"❌ Processing failed with error: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    # Run the async main function
    try:
        exit_code = asyncio.run(run_processing())
        return exit_code
    except KeyboardInterrupt:
        print("\n🛑 Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
