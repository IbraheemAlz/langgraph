"""
Batch Processor for Multi-Agent AI Hiring System
===============================================
Pure batch processing - takes a CSV batch, processes candidates, stores results in JSON.
No visualization - just clean data processing and storage.
"""

import pandas as pd
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.main import create_hiring_workflow
from src.rate_limiter import set_rate_limit
from key_manager import get_key_usage_stats, initialize_api_key_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)


def load_dataset(csv_path: str, max_rows: int = None) -> pd.DataFrame:
    """Load and validate dataset from CSV."""
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"📊 Loaded dataset: {len(df)} candidates from {csv_path}")
        
        # Validate required columns
        required_cols = ['ID', 'Role', 'Job_Description', 'Transcript', 'Resume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Limit rows if specified
        if max_rows and max_rows < len(df):
            df = df.head(max_rows)
            logger.info(f"🔢 Processing first {max_rows} candidates")
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Error loading dataset: {e}")
        raise


def process_candidate(workflow, candidate_data: dict, candidate_num: int, total: int, dataset_index: int = 0) -> dict:
    """Process a single candidate and return results."""
    candidate_id = candidate_data['ID']
    role = candidate_data['Role']
    
    print(f"\n📋 Processing Candidate {candidate_num}/{total}")
    print(f"🆔 ID: {candidate_id}")
    print(f"🎯 Role: {role}")
    print("-" * 50)
    
    logger.info(f"🔄 Starting evaluation for {candidate_id}")
    
    try:
        # Configure workflow for this candidate
        config = {"configurable": {"thread_id": f"candidate_{candidate_id}_{candidate_num}"}}
        
        # Run the workflow
        result = workflow.invoke(candidate_data, config)
        
        # Extract core results from workflow response
        final_decision = result.get('decision', 'unknown')
        bias_classification = result.get('bias_classification', 'unknown') 
        re_evaluation_count = result.get('re_evaluation_count', 0)  # Get actual re-evaluation count
        evaluation_insights = result.get('evaluation_insights', [])
        
        print(f"  ✅ Result: {final_decision}")
        if re_evaluation_count > 0:
            print(f"  ⚠️  Bias detected - {re_evaluation_count} re-evaluation(s)")
        
        # Display evaluation insights
        if evaluation_insights:
            print(f"  📊 Evaluation History:")
            for insight in evaluation_insights:
                eval_type = "re-eval" if insight.get("is_re_evaluation") else "initial"
                classification = insight.get("classification", "pending")
                print(f"    • {eval_type} #{insight['evaluation_number']}: {insight['decision']} → {classification}")
        
        logger.info(f"✅ Completed evaluation for {candidate_id}: {final_decision}")
        
        # Create clean result record
        result_record = {
            "candidate_id": candidate_id,
            "dataset_index": dataset_index,
            "role": role,
            "final_decision": final_decision,
            "bias_classification": bias_classification,
            "re_evaluation_count": re_evaluation_count,
            "evaluation_insights": evaluation_insights,
            "re_evaluation_count": re_evaluation_count,
            "processing_time": datetime.now().isoformat(),
            "workflow_completed": True,
            "job_feedback_count": 1,  # Always 1 for initial evaluation
            "bias_feedback_count": 1 + re_evaluation_count  # 1 initial + re-evaluations
        }
        
        # Include ground truth if available
        if 'decision' in candidate_data:
            result_record['ground_truth_decision'] = candidate_data['decision']
        if 'classification' in candidate_data:
            result_record['ground_truth_bias'] = candidate_data['classification']
            
        return result_record
        
    except Exception as e:
        logger.error(f"❌ Error processing {candidate_id}: {e}")
        
        # Return error record
        # Create clean error record
        error_record = {
            "candidate_id": candidate_id,
            "dataset_index": dataset_index,
            "role": role,
            "final_decision": "error",
            "bias_classification": "error",
            "re_evaluation_count": 0,
            "evaluation_insights": [],
            "processing_time": datetime.now().isoformat(),
            "workflow_completed": False,
            "job_feedback_count": 0,
            "bias_feedback_count": 0,
            "error": str(e)
        }
        
        # Include ground truth if available
        if 'decision' in candidate_data:
            error_record['ground_truth_decision'] = candidate_data['decision']
        if 'classification' in candidate_data:
            error_record['ground_truth_bias'] = candidate_data['classification']
        
        print(f"  ❌ Error: {e}")
        return error_record


def save_results(results: list, output_path: str = "results/json/batch_results.json"):
    """Save results to JSON file with metadata."""
    
    # Ensure results directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate summary statistics
    total_candidates = len(results)
    successful = len([r for r in results if r['workflow_completed'] == True])
    errors = total_candidates - successful
    
    # Create output data
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_candidates": total_candidates,
            "successful_evaluations": successful,
            "errors": errors,
            "success_rate": (successful / total_candidates * 100) if total_candidates > 0 else 0,
            "version": "batch_processor_v1.0"
        },
        "results": results
    }
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 Results saved to: {output_path}")
    return output_path


def save_incremental_result(result: dict, output_path: str) -> None:
    """Save a single result incrementally to JSON file."""
    # Ensure results directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing results if file exists
    if Path(output_path).exists():
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            existing_results = data.get("results", [])
        except (json.JSONDecodeError, KeyError):
            existing_results = []
    else:
        existing_results = []
    
    # Add new result
    existing_results.append(result)
    
    # Calculate updated statistics
    total_candidates = len(existing_results)
    successful = len([r for r in existing_results if r['workflow_completed'] == True])
    errors = total_candidates - successful
    
    # Create updated output data
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_candidates": total_candidates,
            "successful_evaluations": successful,
            "errors": errors,
            "success_rate": (successful / total_candidates * 100) if total_candidates > 0 else 0,
            "version": "batch_processor_v1.0",
            "incremental_save": True
        },
        "results": existing_results
    }
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.debug(f"💾 Incremental save: {len(existing_results)} results in {output_path}")


def load_existing_results(output_path: str) -> list:
    """Load existing results from JSON file."""
    if Path(output_path).exists():
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get("results", [])
        except (json.JSONDecodeError, KeyError):
            logger.warning(f"Could not load existing results from {output_path}")
            return []
    return []


def print_summary(results: list):
    """Print processing summary."""
    total = len(results)
    successful = len([r for r in results if r['workflow_completed'] == True])
    errors = total - successful
    
    # Decision statistics (successful only)
    success_results = [r for r in results if r['workflow_completed'] == True]
    selected = len([r for r in success_results if r['final_decision'] == 'select'])
    rejected = len([r for r in success_results if r['final_decision'] == 'reject'])
    
    # Bias statistics
    biased = len([r for r in success_results if r['bias_classification'] == 'biased'])
    total_reevals = sum([r['re_evaluation_count'] for r in success_results])
    
    print("\n" + "="*80)
    print("📊 BATCH PROCESSING SUMMARY")
    print("="*80)
    
    print("📈 PROCESSING STATISTICS:")
    print("-"*50)
    print(f"📊 Total Candidates: {total}")
    print(f"✅ Successful Evaluations: {successful}")
    print(f"❌ Errors: {errors}")
    print(f"📊 Success Rate: {(successful/total*100):.1f}%")
    
    if success_results:
        print("\n📋 DECISION STATISTICS:")
        print("-"*30)
        print(f"👍 Selected: {selected} ({selected/successful*100:.1f}%)")
        print(f"👎 Rejected: {rejected} ({rejected/successful*100:.1f}%)")
        
        print("\n🔍 BIAS ANALYSIS:")
        print("-"*20)
        print(f"⚠️  Bias Detected: {biased} ({biased/successful*100:.1f}%)")
        print(f"🔄 Total Re-evaluations: {total_reevals}")
        print(f"📊 Avg Re-evaluations: {total_reevals/successful:.2f}")
    
    # Show API key usage statistics
    try:
        print("\n🔑 API KEY USAGE STATISTICS:")
        print("-"*30)
        stats = get_key_usage_stats()
        total_api_requests = sum(stat['total_requests'] for stat in stats.values())
        active_keys = len([stat for stat in stats.values() if stat['total_requests'] > 0])
        
        print(f"🔑 Active API Keys: {active_keys}/{len(stats)}")
        print(f"📈 Total API Requests: {total_api_requests}")
        
        for key_id, stat in stats.items():
            if stat['total_requests'] > 0:
                print(f"  • {key_id}: {stat['total_requests']} requests")
    except Exception as e:
        print(f"\n⚠️  Could not load API key statistics: {e}")


def main():
    """Main batch processing function."""
    parser = argparse.ArgumentParser(description="Process hiring candidates in batch")
    parser.add_argument("--csv", default="sample-data.csv", help="CSV file path")
    parser.add_argument("--max-rows", type=int, help="Maximum rows to process")
    parser.add_argument("--output", default="results/json/batch_results.json", help="Output JSON file")
    parser.add_argument("--rate-limit", type=int, default=5, help="Requests per minute")
    parser.add_argument("--no-incremental", action="store_true", help="Disable incremental saving (save only at end)")
    parser.add_argument("--clear-existing", action="store_true", help="Clear existing results and start fresh")
    args = parser.parse_args()
    
    start_time = datetime.now()
    
    print("🚀 Starting Batch Processing")
    print(f"⏱️  Rate limit: {args.rate_limit} requests per minute")
    print("🛡️ Comprehensive error handling enabled")
    
    if not args.no_incremental:
        print("💾 Incremental saving enabled - results saved after each candidate")
    else:
        print("💾 Batch saving mode - results saved only at completion")
    
    if args.clear_existing:
        print("🗑️  Clear existing results mode enabled")
    
    try:
        # Setup rate limiting with API Key Manager
        initialize_api_key_manager(rate_limit_per_minute=args.rate_limit)
        print(f"🔑 API Key Manager initialized with {args.rate_limit} requests per minute per key")
        
        # Legacy rate limiter call (now just shows warning)
        set_rate_limit(args.rate_limit)
        
        # Load dataset
        df = load_dataset(args.csv, args.max_rows)
        
        # Create workflow (rate limiter is used globally)
        workflow = create_hiring_workflow()
        logger.info("🏗️ Hiring workflow created successfully")
        
        print("\n" + "="*80)
        print("🚀 STARTING BATCH PROCESSING OF HIRING DECISIONS")
        print("="*80)
        
        # Handle existing results
        existing_results = []
        processed_candidates = set()
        
        if args.clear_existing and Path(args.output).exists():
            Path(args.output).unlink()
            print("🗑️  Cleared existing results file")
        
        if not args.no_incremental and not args.clear_existing:
            existing_results = load_existing_results(args.output)
            
            if existing_results:
                processed_candidates = {r.get('candidate_id', '') for r in existing_results}
                print(f"📂 Found existing results: {len(existing_results)} candidates already processed")
                print(f"🔄 Will skip already processed candidates and continue from where left off")
        
        # Process all candidates
        results = existing_results.copy() if not args.no_incremental else []
        total_candidates = len(df)
        new_processed = 0
        
        for idx, row in df.iterrows():
            candidate_data = row.to_dict()
            candidate_id = candidate_data.get('candidate_id', f'candidate_{idx}')
            
            # Skip if already processed (only when incremental mode is enabled)
            if not args.no_incremental and candidate_id in processed_candidates:
                print(f"⏭️  Skipping {candidate_id} (already processed)")
                continue
            
            # Process candidate
            result = process_candidate(workflow, candidate_data, idx + 1, total_candidates, idx)
            results.append(result)
            new_processed += 1
            
            # Save incrementally after each candidate (unless disabled)
            if not args.no_incremental:
                save_incremental_result(result, args.output)
        
        # Final save to ensure everything is captured
        output_file = save_results(results, args.output)
        
        if not args.no_incremental and new_processed > 0:
            print(f"\n✨ Processed {new_processed} new candidates (total: {len(results)})")
        
        # Print summary
        print_summary(results)
        
        # Final stats
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n⏱️  Total processing time: {duration}")
        print("🎉 Batch processing completed!")
        print(f"📁 Results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"❌ Batch processing failed: {e}")
        print(f"\n❌ Batch processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
