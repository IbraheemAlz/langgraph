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
        
    async def process_batch_file(self, input_file: str, job_requirements: Dict[str, Any], output_file: str = None, force_reprocess: bool = False):
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
            
            self.logger.info(f"📊 Loaded {len(candidates)} candidates from CSV")
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
        
        # Initialize JSON file structure immediately
        timestamp = int(time.time())
        if output_file is None:
            output_file = f"{self.config.RESULTS_FOLDER}/json/runpod_batch_results_{timestamp}.json"
        
        self.logger.info(f"🎯 Target output file: {output_file}")
        
        # 🔄 NEW: Load existing results and filter candidates
        processed_ids = self._load_existing_results(output_file)
        original_count = len(candidates)
        
        if force_reprocess:
            self.logger.info("🔥 FORCE MODE: Processing all candidates (ignoring existing results)")
            candidates = candidates  # Process all candidates
            existing_results = []
        else:
            candidates = self._filter_unprocessed_candidates(candidates, processed_ids)
            
            if len(candidates) == 0:
                self.logger.info("🎉 All candidates have already been processed!")
                self.logger.info(f"📂 Results are in: {output_file}")
                return output_file
            
            # Load existing results for continuation
            existing_results = []
            if processed_ids:
                try:
                    if Path(output_file).exists():
                        with open(output_file, 'r', encoding='utf-8') as f:
                            existing_data = json.load(f)
                            existing_results = existing_data.get('results', [])
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not load existing results: {e}")
                    existing_results = []
        
        all_results = existing_results.copy()  # Start with existing results
        
        # Create initial JSON structure
        initial_output_data = {
            "metadata": {
                "total_candidates": original_count,  # Use original count from CSV
                "successful_analyses": len([r for r in existing_results if r.get('error') is None]),
                "failed_analyses": len([r for r in existing_results if r.get('error') is not None]),
                "processing_time_seconds": 0,
                "average_rate_per_second": 0,
                "job_requirements": job_requirements,
                "model_used": self.config.MODEL_NAME,
                "pod_id": self.config.RUNPOD_POD_ID,
                "batch_size": batch_size,
                "concurrent_requests": self.concurrent_limit,
                "timestamp": time.time(),
                "status": "force_reprocessing" if force_reprocess else ("resuming" if existing_results else "processing"),
                "batches_completed": 0,
                "batches_total": (len(candidates) + batch_size - 1) // batch_size,
                "already_processed": 0 if force_reprocess else len(existing_results),
                "remaining_to_process": len(candidates)
            },
            "results": existing_results,  # Include existing results
            "summary": {
                "total_processed": len(existing_results),
                "success_rate": (len([r for r in existing_results if r.get('error') is None]) / len(existing_results) * 100) if existing_results else 0,
                "avg_processing_time_per_candidate": 0
            }
        }
        
        # Create/update the JSON file
        status = "FORCE" if force_reprocess else ("RESUME" if existing_results else "INITIAL")
        self._save_json_file(output_file, initial_output_data, status)
        
        self.logger.info(f"⚡ Processing with {self.concurrent_limit} concurrent requests")
        self.logger.info(f"📦 Batch size: {batch_size} candidates per batch")
        
        if existing_results:
            self.logger.info(f"🔄 RESUMING processing from {len(existing_results)} existing results")
            self.logger.info(f"📊 Remaining: {len(candidates)} candidates to process")
        
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
                
                # Update JSON file after each batch
                await self._update_json_after_batch(output_file, all_results, candidates, job_requirements, 
                                                   batch_num, total_batches, start_time, timestamp)
                
                # Progress metrics
                batch_time = time.time() - batch_start
                processed = len(all_results)
                total_elapsed = time.time() - start_time
                rate = len(candidates) / total_elapsed if total_elapsed > 0 else 0  # Rate for new candidates only
                remaining = len(candidates) - (len(all_results) - len(existing_results))
                eta = remaining / rate if rate > 0 else 0
                
                self.logger.info(f"✅ Batch {batch_num} complete in {batch_time:.1f}s")
                self.logger.info(f"📈 Progress: {processed}/{original_count} total ({processed - len(existing_results)}/{len(candidates)} new) ({rate:.1f} candidates/sec)")
                self.logger.info(f"⏱️ ETA: {eta:.0f}s remaining")
                self.logger.info(f"💾 JSON updated: {output_file}")
                
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
                
                # Still update JSON even with errors
                await self._update_json_after_batch(output_file, all_results, candidates, job_requirements, 
                                                   batch_num, total_batches, start_time, timestamp)
        
        # Final update to mark completion
        await self._finalize_json_file(output_file, all_results, candidates, job_requirements, start_time, timestamp)
        
        # Final summary
        total_time = time.time() - start_time
        successful_results = [r for r in all_results if r.get('error') is None]
        failed_results = [r for r in all_results if r.get('error') is not None]
        new_results = all_results[len(existing_results):]  # Only new results processed in this session
        new_successful = [r for r in new_results if r.get('error') is None]
        new_failed = [r for r in new_results if r.get('error') is not None]
        
        self.logger.info("🎉 Batch processing complete!")
        self.logger.info(f"📊 Final Results Summary:")
        self.logger.info(f"  • Total candidates (CSV): {original_count}")
        self.logger.info(f"  • Already processed: {len(existing_results)}")
        self.logger.info(f"  • Processed this session: {len(new_results)}")
        self.logger.info(f"  • New successful: {len(new_successful)}")
        self.logger.info(f"  • New failed: {len(new_failed)}")
        self.logger.info(f"  • Overall success rate: {len(successful_results)/len(all_results)*100:.1f}%")
        self.logger.info(f"  • Session processing time: {total_time:.1f}s")
        if len(new_results) > 0:
            self.logger.info(f"  • Session processing rate: {len(new_results)/total_time:.1f} candidates/sec")
        self.logger.info(f"💾 Final results saved to: {output_file}")
        
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

    def _save_json_file(self, file_path: str, data: Dict[str, Any], status: str):
        """Saves the current state of the processing to a JSON file."""
        try:
            # Ensure the directory exists
            file_path_obj = Path(file_path)
            file_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"💾 Saving JSON file: {file_path} (Status: {status})")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Verify file was saved successfully
            if file_path_obj.exists() and file_path_obj.stat().st_size > 0:
                file_size = file_path_obj.stat().st_size
                result_count = len(data.get('results', []))
                self.logger.info(f"✅ JSON file saved: {file_path} ({file_size:,} bytes, {result_count} results)")
            else:
                self.logger.error(f"❌ JSON file not created or empty: {file_path}")
                
        except PermissionError as e:
            self.logger.warning(f"🚫 Permission denied for saving JSON: {e}")
            # Try fallback location
            fallback_path = f"/tmp/runpod_batch_results_{int(time.time())}.json"
            try:
                with open(fallback_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                self.logger.info(f"✅ JSON saved to fallback location: {fallback_path}")
            except Exception as fallback_e:
                self.logger.error(f"❌ Fallback save also failed: {fallback_e}")
        except OSError as e:
            self.logger.warning(f"💾 OS error for saving JSON: {e}")
        except Exception as e:
            self.logger.error(f"❌ Failed to save JSON file {file_path}: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())

    async def _update_json_after_batch(self, file_path: str, all_results: List[Dict[str, Any]], candidates: List[Dict[str, Any]], 
                                        job_requirements: Dict[str, Any], batch_num: int, total_batches: int, 
                                        start_time: float, timestamp: int):
        """Updates the JSON file with results from the current batch."""
        # Load the original count and existing results count
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                original_count = existing_data.get('metadata', {}).get('total_candidates', len(all_results))
                already_processed = existing_data.get('metadata', {}).get('already_processed', 0)
        except:
            original_count = len(all_results)
            already_processed = 0
        
        processed = len(all_results)
        successful = [r for r in all_results if r.get('error') is None]
        failed = [r for r in all_results if r.get('error') is not None]
        
        current_output_data = {
            "metadata": {
                "total_candidates": original_count,
                "successful_analyses": len(successful),
                "failed_analyses": len(failed),
                "processing_time_seconds": time.time() - start_time,
                "average_rate_per_second": (processed - already_processed) / (time.time() - start_time) if (time.time() - start_time) > 0 else 0,
                "job_requirements": job_requirements,
                "model_used": self.config.MODEL_NAME,
                "pod_id": self.config.RUNPOD_POD_ID,
                "batch_size": self.config.BATCH_SIZE,
                "concurrent_requests": self.concurrent_limit,
                "timestamp": time.time(),
                "status": "processing",
                "batches_completed": batch_num,
                "batches_total": total_batches,
                "already_processed": already_processed,
                "remaining_to_process": len(candidates) - (processed - already_processed)
            },
            "results": all_results,
            "summary": {
                "total_processed": processed,
                "success_rate": len(successful) / processed * 100 if processed > 0 else 0,
                "avg_processing_time_per_candidate": (time.time() - start_time) / (processed - already_processed) if (processed - already_processed) > 0 else 0
            }
        }
        self._save_json_file(file_path, current_output_data, f"BATCH_{batch_num}")

    async def _finalize_json_file(self, file_path: str, all_results: List[Dict[str, Any]], candidates: List[Dict[str, Any]], 
                                   job_requirements: Dict[str, Any], start_time: float, timestamp: int):
        """Finalizes the JSON file by marking completion and adding final metrics."""
        # Load the original count and existing results count
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                original_count = existing_data.get('metadata', {}).get('total_candidates', len(all_results))
                already_processed = existing_data.get('metadata', {}).get('already_processed', 0)
        except:
            original_count = len(all_results)
            already_processed = 0
            
        processed = len(all_results)
        successful = [r for r in all_results if r.get('error') is None]
        failed = [r for r in all_results if r.get('error') is not None]
        
        final_output_data = {
            "metadata": {
                "total_candidates": original_count,
                "successful_analyses": len(successful),
                "failed_analyses": len(failed),
                "processing_time_seconds": time.time() - start_time,
                "average_rate_per_second": (processed - already_processed) / (time.time() - start_time) if (time.time() - start_time) > 0 else 0,
                "job_requirements": job_requirements,
                "model_used": self.config.MODEL_NAME,
                "pod_id": self.config.RUNPOD_POD_ID,
                "batch_size": self.config.BATCH_SIZE,
                "concurrent_requests": self.concurrent_limit,
                "timestamp": time.time(),
                "status": "completed",
                "batches_completed": (len(candidates) + self.config.BATCH_SIZE - 1) // self.config.BATCH_SIZE,
                "batches_total": (len(candidates) + self.config.BATCH_SIZE - 1) // self.config.BATCH_SIZE,
                "already_processed": already_processed,
                "session_processed": processed - already_processed
            },
            "results": all_results,
            "summary": {
                "total_processed": processed,
                "success_rate": len(successful) / processed * 100 if processed > 0 else 0,
                "avg_processing_time_per_candidate": (time.time() - start_time) / (processed - already_processed) if (processed - already_processed) > 0 else 0
            }
        }
        self._save_json_file(file_path, final_output_data, "FINAL")

    def _load_existing_results(self, output_file: str) -> List[str]:
        """Load existing results and return list of processed candidate IDs"""
        processed_ids = set()
        
        # Check if the specified output file exists
        if Path(output_file).exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    results = existing_data.get('results', [])
                    for result in results:
                        candidate_id = result.get('candidate_id')
                        if candidate_id:
                            processed_ids.add(candidate_id)
                    
                self.logger.info(f"📂 Found existing output file: {output_file}")
                self.logger.info(f"✅ Already processed: {len(processed_ids)} candidates")
                return list(processed_ids)
            except Exception as e:
                self.logger.warning(f"⚠️ Could not read existing file {output_file}: {e}")
        
        # Also check for other JSON files in results folder
        results_dir = Path(self.config.RESULTS_FOLDER) / "json"
        if results_dir.exists():
            json_files = list(results_dir.glob("*.json"))
            if json_files:
                self.logger.info(f"🔍 Found {len(json_files)} existing JSON files in results folder")
                
                for json_file in json_files:
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            results = data.get('results', [])
                            for result in results:
                                candidate_id = result.get('candidate_id')
                                if candidate_id:
                                    processed_ids.add(candidate_id)
                    except Exception as e:
                        self.logger.warning(f"⚠️ Could not read {json_file}: {e}")
                        continue
                
                if processed_ids:
                    self.logger.info(f"✅ Total processed candidates found: {len(processed_ids)}")
                    # Show some examples
                    sample_ids = list(processed_ids)[:5]
                    self.logger.info(f"📋 Sample processed IDs: {sample_ids}")
        
        return list(processed_ids)
    
    def _filter_unprocessed_candidates(self, candidates: List[Dict[str, Any]], processed_ids: List[str]) -> List[Dict[str, Any]]:
        """Filter out candidates that have already been processed"""
        if not processed_ids:
            return candidates
        
        processed_set = set(processed_ids)
        unprocessed = []
        skipped_count = 0
        
        for candidate in candidates:
            candidate_id = candidate.get('id') or candidate.get('ID')
            if candidate_id and candidate_id in processed_set:
                skipped_count += 1
            else:
                unprocessed.append(candidate)
        
        self.logger.info(f"🔄 Filtering results:")
        self.logger.info(f"   📊 Total candidates: {len(candidates)}")
        self.logger.info(f"   ✅ Already processed: {skipped_count}")
        self.logger.info(f"   🔄 Remaining to process: {len(unprocessed)}")
        
        return unprocessed

def test_file_creation():
    """Test function to verify file creation works in the environment"""
    import os
    import json
    from pathlib import Path
    
    print("🧪 Testing file creation in RunPod environment...")
    
    # Test data
    test_data = {
        "test": True,
        "timestamp": time.time(),
        "environment": {
            "cwd": os.getcwd(),
            "pod_id": os.environ.get('RUNPOD_POD_ID', 'local'),
            "workspace": os.environ.get('WORKSPACE_PATH', 'unknown')
        }
    }
    
    # Test locations
    timestamp = int(time.time())
    test_locations = [
        f"./test_file_{timestamp}.json",
        f"/workspace/test_file_{timestamp}.json",
        f"/workspace/langgraph/test_file_{timestamp}.json", 
        f"/tmp/test_file_{timestamp}.json",
        f"results/test_file_{timestamp}.json"
    ]
    
    success_count = 0
    for location in test_locations:
        try:
            # Create directory if needed
            Path(location).parent.mkdir(parents=True, exist_ok=True)
            
            # Write test file
            with open(location, 'w') as f:
                json.dump(test_data, f, indent=2)
            
            # Verify file
            if Path(location).exists() and Path(location).stat().st_size > 0:
                print(f"✅ SUCCESS: {location}")
                success_count += 1
                # Clean up test file
                try:
                    Path(location).unlink()
                except:
                    pass
            else:
                print(f"❌ FAILED: {location} (file not created or empty)")
                
        except Exception as e:
            print(f"❌ FAILED: {location} - {e}")
    
    print(f"📊 File creation test results: {success_count}/{len(test_locations)} locations successful")
    return success_count > 0

def main():
    """CLI interface for RunPod batch processing"""
    parser = argparse.ArgumentParser(description="RunPod Batch Processor for AI Hiring System")
    parser.add_argument("--input", help="Input CSV file path")
    parser.add_argument("--output", help="Output JSON file path (optional)")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--job-title", default="Software Engineer", help="Job title")
    parser.add_argument("--required-skills", default="Python,JavaScript,AI,Machine Learning", help="Required skills (comma-separated)")
    parser.add_argument("--experience-level", default="Mid-level", help="Experience level")
    parser.add_argument("--education", default="Bachelor's degree preferred", help="Education requirements")
    parser.add_argument("--test", action="store_true", help="Run file creation test only")
    parser.add_argument("--force", action="store_true", help="Force processing all candidates (skip resume functionality)")
    
    args = parser.parse_args()
    
    # If test mode, run test and exit
    if args.test:
        return 0 if test_file_creation() else 1
    
    # Validate input file (only required if not in test mode)
    if not args.input:
        print("❌ --input argument is required (unless using --test)")
        return 1
        
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
                args.output,
                force_reprocess=args.force
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
