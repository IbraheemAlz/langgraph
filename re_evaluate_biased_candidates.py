#!/usr/bin/env python3
"""
Re-evaluate Biased Candidates Script
Finds candidates marked as biased but never re-evaluated and processes them through the workflow
"""

import json
import asyncio
import aiohttp
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
import time
import sys
import pandas as pd

# Add src to path
sys.path.append('src')
from src.config import Config

class BiasedCandidateReEvaluator:
    """Re-evaluates biased candidates using the proper workflow API"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.config = Config()
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for re-evaluation process"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f'{self.config.RESULTS_FOLDER}/re_evaluation.log')
            ]
        )
        return logging.getLogger(__name__)
    
    def find_biased_candidates(self, results_folder: str = None, csv_file: str = None) -> List[Dict[str, Any]]:
        """Find candidates that are biased but never re-evaluated and match with original CSV data"""
        if results_folder is None:
            results_folder = f"{self.config.RESULTS_FOLDER}/json"
        
        results_dir = Path(results_folder)
        if not results_dir.exists():
            self.logger.error(f"Results folder not found: {results_folder}")
            return []
        
        # Load original CSV data if provided
        csv_data = {}
        if csv_file and Path(csv_file).exists():
            try:
                df = pd.read_csv(csv_file)
                # Create lookup by ID
                for _, row in df.iterrows():
                    candidate_id = str(row.get('ID', row.get('id', '')))
                    if candidate_id:
                        csv_data[candidate_id] = row.to_dict()
                self.logger.info(f"📊 Loaded {len(csv_data)} candidates from CSV: {csv_file}")
            except Exception as e:
                self.logger.warning(f"Could not load CSV file {csv_file}: {e}")
        
        biased_candidates = []
        json_files = list(results_dir.glob("*.json"))
        
        self.logger.info(f"🔍 Scanning {len(json_files)} JSON files for biased candidates...")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    results = data.get('results', [])
                    
                    for result in results:
                        # Check if candidate is biased and never re-evaluated
                        bias_classification = result.get('bias_classification', 'unbiased')
                        re_evaluation_count = result.get('re_evaluation_count', 0)
                        candidate_id = result.get('candidate_id', 'unknown')
                        
                        if bias_classification == 'biased' and re_evaluation_count == 0:
                            # Try to get original data from CSV
                            if candidate_id in csv_data:
                                csv_row = csv_data[candidate_id]
                                candidate_data = {
                                    'id': candidate_id,
                                    'Resume': csv_row.get('Resume', ''),
                                    'Job_Description': csv_row.get('Job_Description', ''),
                                    'Transcript': csv_row.get('Transcript', ''),
                                    'Role': csv_row.get('Role', result.get('role', 'Unknown Role')),
                                    'dataset_index': result.get('dataset_index', 0)
                                }
                            else:
                                # Fallback when no CSV data available
                                candidate_data = {
                                    'id': candidate_id,
                                    'Resume': '[Resume data not available - requires original CSV]',
                                    'Job_Description': '[Job_Description data not available - requires original CSV]',
                                    'Transcript': '[Transcript data not available - requires original CSV]',
                                    'Role': result.get('role', 'Unknown Role'),
                                    'dataset_index': result.get('dataset_index', 0)
                                }
                            
                            # Add metadata for tracking
                            candidate_data['_original_result'] = result
                            candidate_data['_source_file'] = str(json_file)
                            candidate_data['_has_csv_data'] = candidate_id in csv_data
                            
                            biased_candidates.append(candidate_data)
                            self.logger.debug(f"Found biased candidate: {candidate_id} (CSV data: {'✅' if candidate_id in csv_data else '❌'})")
                            
            except Exception as e:
                self.logger.warning(f"Could not read {json_file}: {e}")
                continue
        
        candidates_with_data = len([c for c in biased_candidates if c.get('_has_csv_data', False)])
        candidates_without_data = len(biased_candidates) - candidates_with_data
        
        self.logger.info(f"✅ Found {len(biased_candidates)} biased candidates to re-evaluate")
        if csv_file:
            self.logger.info(f"📊 With CSV data: {candidates_with_data}, Without CSV data: {candidates_without_data}")
            if candidates_without_data > 0:
                self.logger.warning(f"⚠️ {candidates_without_data} candidates cannot be re-evaluated without original CSV data")
        
        return biased_candidates
    
    def _extract_from_insights(self, result: Dict[str, Any], field: str) -> str:
        """Try to extract original input data from various places in the result"""
        # The JSON results don't contain the full original data
        # This is why we need the original CSV file
        return f"[{field} data not available in JSON - requires original CSV]"
    
    async def re_evaluate_candidate(self, candidate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Re-evaluate a single candidate using the API with proper workflow"""
        candidate_id = candidate_data.get('id', 'unknown')
        
        try:
            timeout = aiohttp.ClientTimeout(total=self.config.REQUEST_TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Prepare request data
                request_data = {
                    "candidate_data": {
                        'id': candidate_data['id'],
                        'Resume': candidate_data['Resume'],
                        'Job_Description': candidate_data['Job_Description'],
                        'Transcript': candidate_data['Transcript'],
                        'Role': candidate_data['Role'],
                        'dataset_index': candidate_data.get('dataset_index', 0)
                    },
                    "job_requirements": {
                        "title": candidate_data['Role'],
                        "required_skills": ["Various skills"],
                        "experience_level": "Mid-level",
                        "education_requirements": "Bachelor's degree preferred"
                    }
                }
                
                async with session.post(
                    f"{self.api_url}/analyze_candidate",
                    json=request_data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Log the re-evaluation result
                        final_decision = result.get('final_decision', 'unknown')
                        bias_classification = result.get('bias_classification', 'unknown')
                        re_evaluation_count = result.get('re_evaluation_count', 0)
                        
                        self.logger.info(f"✅ Re-evaluated {candidate_id}: {final_decision} (bias: {bias_classification}, re-evals: {re_evaluation_count})")
                        
                        return result
                    else:
                        error_text = await response.text()
                        raise Exception(f"API error {response.status}: {error_text}")
                        
        except Exception as e:
            self.logger.error(f"❌ Failed to re-evaluate {candidate_id}: {e}")
            return {
                "candidate_id": candidate_id,
                "error": str(e),
                "job_match": None,
                "bias_analysis": None,
                "timestamp": time.time()
            }
    
    async def re_evaluate_batch(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Re-evaluate a batch of biased candidates"""
        self.logger.info(f"🔄 Starting re-evaluation of {len(candidates)} biased candidates...")
        
        # Process with limited concurrency
        semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent requests
        
        async def process_single(candidate):
            async with semaphore:
                return await self.re_evaluate_candidate(candidate)
        
        # Create tasks for all candidates
        tasks = [process_single(candidate) for candidate in candidates]
        
        # Process with progress tracking
        results = []
        completed = 0
        
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
            completed += 1
            
            if completed % 10 == 0 or completed == len(candidates):
                self.logger.info(f"📈 Progress: {completed}/{len(candidates)} candidates re-evaluated")
        
        return results
    
    def save_re_evaluation_results(self, results: List[Dict[str, Any]], output_file: str = None):
        """Save re-evaluation results to a new JSON file"""
        if output_file is None:
            timestamp = int(time.time())
            output_file = f"{self.config.RESULTS_FOLDER}/json/re_evaluation_results_{timestamp}.json"
        
        # Prepare output data
        successful_results = [r for r in results if r.get('error') is None]
        failed_results = [r for r in results if r.get('error') is not None]
        
        output_data = {
            "metadata": {
                "type": "bias_re_evaluation",
                "total_candidates": len(results),
                "successful_re_evaluations": len(successful_results),
                "failed_re_evaluations": len(failed_results),
                "timestamp": time.time(),
                "success_rate": len(successful_results) / len(results) * 100 if results else 0
            },
            "results": results,
            "summary": {
                "re_evaluation_outcomes": {
                    "now_unbiased": len([r for r in successful_results if r.get('bias_classification') == 'unbiased']),
                    "still_biased": len([r for r in successful_results if r.get('bias_classification') == 'biased']),
                    "decision_changes": len([r for r in successful_results if self._decision_changed(r)])
                }
            }
        }
        
        try:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"💾 Re-evaluation results saved to: {output_file}")
            self.logger.info(f"📊 Summary: {len(successful_results)} successful, {len(failed_results)} failed")
            
            return output_file
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save results: {e}")
            return None
    
    def _decision_changed(self, result: Dict[str, Any]) -> bool:
        """Check if the decision changed from original"""
        original_result = result.get('_original_result', {})
        original_decision = original_result.get('final_decision', 'unknown')
        new_decision = result.get('final_decision', 'unknown')
        return original_decision != new_decision
    
    async def health_check(self):
        """Check if the API service is healthy"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/health", timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        health_data = await response.json()
                        self.logger.info(f"✅ API service healthy")
                        return True
                    else:
                        self.logger.error(f"❌ API service unhealthy: {response.status}")
                        return False
        except Exception as e:
            self.logger.error(f"❌ Health check failed: {e}")
            return False

async def main():
    """Main function for the re-evaluation script"""
    parser = argparse.ArgumentParser(description="Re-evaluate Biased Candidates")
    parser.add_argument("--results-folder", help="Path to results folder containing JSON files")
    parser.add_argument("--csv-file", help="Path to original CSV file with candidate data (required for re-evaluation)")
    parser.add_argument("--output", help="Output file for re-evaluation results")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--max-candidates", type=int, help="Maximum number of candidates to re-evaluate")
    parser.add_argument("--dry-run", action="store_true", help="Only find candidates, don't re-evaluate")
    parser.add_argument("--skip-missing-data", action="store_true", help="Skip candidates without CSV data instead of failing")
    
    args = parser.parse_args()
    
    # Initialize re-evaluator
    evaluator = BiasedCandidateReEvaluator(api_url=args.api_url)
    
    # Find biased candidates
    biased_candidates = evaluator.find_biased_candidates(args.results_folder, args.csv_file)
    
    if not biased_candidates:
        evaluator.logger.info("🎉 No biased candidates found that need re-evaluation!")
        return 0
    
    # Filter out candidates without CSV data if requested
    if args.skip_missing_data:
        candidates_with_data = [c for c in biased_candidates if c.get('_has_csv_data', False)]
        candidates_without_data = len(biased_candidates) - len(candidates_with_data)
        if candidates_without_data > 0:
            evaluator.logger.info(f"🔄 Skipping {candidates_without_data} candidates without CSV data")
        biased_candidates = candidates_with_data
    elif not args.csv_file:
        evaluator.logger.error("❌ CSV file is required for re-evaluation. Use --csv-file or --skip-missing-data")
        return 1
    else:
        # Check if any candidates lack CSV data
        candidates_without_data = [c for c in biased_candidates if not c.get('_has_csv_data', False)]
        if candidates_without_data:
            evaluator.logger.error(f"❌ {len(candidates_without_data)} candidates lack CSV data. Use --skip-missing-data to ignore them")
            return 1
    
    if not biased_candidates:
        evaluator.logger.info("🎉 No biased candidates with data found for re-evaluation!")
        return 0
    
    # Limit candidates if specified
    if args.max_candidates and len(biased_candidates) > args.max_candidates:
        evaluator.logger.info(f"📝 Limiting to first {args.max_candidates} candidates")
        biased_candidates = biased_candidates[:args.max_candidates]
    
    # Show summary
    evaluator.logger.info(f"📋 Found candidates to re-evaluate:")
    for i, candidate in enumerate(biased_candidates[:5]):  # Show first 5
        candidate_id = candidate.get('id', 'unknown')
        evaluator.logger.info(f"  {i+1}. {candidate_id}")
    if len(biased_candidates) > 5:
        evaluator.logger.info(f"  ... and {len(biased_candidates) - 5} more")
    
    # Dry run mode
    if args.dry_run:
        evaluator.logger.info("🔍 Dry run mode - not performing re-evaluation")
        return 0
    
    # Health check
    evaluator.logger.info("🔍 Checking API health...")
    if not await evaluator.health_check():
        evaluator.logger.error("❌ API service not healthy. Please ensure the service is running.")
        evaluator.logger.info("💡 Try running: python run_on_runpod.py")
        return 1
    
    # Re-evaluate candidates
    results = await evaluator.re_evaluate_batch(biased_candidates)
    
    # Save results
    output_file = evaluator.save_re_evaluation_results(results, args.output)
    
    if output_file:
        evaluator.logger.info(f"✅ Re-evaluation complete! Results saved to: {output_file}")
        return 0
    else:
        evaluator.logger.error("❌ Failed to save results")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Re-evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1) 