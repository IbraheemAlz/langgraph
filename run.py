#!/usr/bin/env python3
"""
Entry point for the Multi-Agent AI Hiring System.
This script provides a simple way to run the system from the command line.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

def main():
    """Main entry point for the hiring system."""
    from src.main import create_hiring_workflow
    
    # Create the workflow
    app = create_hiring_workflow()
    
    # Sample data for demonstration - Updated for new state schema
    sample_data = {
        "Resume": "Senior developer with 5 years Python and Django experience, CS degree from top university",
        "Job_Description": "Software Engineer with 3+ years Python experience, Django knowledge preferred",
        "Transcript": "Candidate demonstrated strong technical skills and problem-solving ability. Good communication skills.",
        "Role": "Software Engineer",
        "decision": "",
        "bias_classification": "",
        "re_evaluation_count": 0,
        "feedback": [],
        "evaluation_insights": [],
        "timestamp": "",
        "process_complete": False
    }
    
    print("🚀 Starting Multi-Agent AI Hiring System Demo")
    print("⏱️  Single candidate evaluation")
    print("🛡️ Comprehensive error handling enabled")
    
    print("\n" + "="*80)
    print("🚀 STARTING HIRING DECISION EVALUATION")
    print("="*80)
    
    print("� Processing Candidate 1/1")
    print(f"🆔 ID: demo_candidate")
    print(f"🎯 Role: {sample_data['Role']}")
    print("-" * 50)
    
    # Run the workflow
    config = {"configurable": {"thread_id": "demo_session"}}
    
    try:
        result = app.invoke(sample_data, config)
        
        # Extract results (matching batch_processor format)
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
        
        print(f"  ✅ Result: {final_decision}")
        if re_evaluation_count > 0:
            print(f"  ⚠️  Bias detected - {re_evaluation_count} re-evaluation(s)")
        
        print("\n" + "="*80)
        print("📊 EVALUATION SUMMARY")
        print("="*80)
        
        print("📈 PROCESSING STATISTICS:")
        print("-"*50)
        print(f"📊 Total Candidates: 1")
        print(f"✅ Successful Evaluations: 1")
        print(f"❌ Errors: 0")
        print(f"📊 Success Rate: 100.0%")
        
        print("\n📋 DECISION STATISTICS:")
        print("-"*30)
        if final_decision == 'select':
            print(f"👍 Selected: 1 (100.0%)")
            print(f"👎 Rejected: 0 (0.0%)")
        else:
            print(f"👍 Selected: 0 (0.0%)")
            print(f"👎 Rejected: 1 (100.0%)")
        
        print("\n🔍 BIAS ANALYSIS:")
        print("-"*20)
        bias_detected = 1 if bias_classification == 'biased' else 0
        print(f"⚠️  Bias Detected: {bias_detected} ({bias_detected*100:.1f}%)")
        print(f"� Total Re-evaluations: {re_evaluation_count}")
        print(f"� Avg Re-evaluations: {re_evaluation_count:.2f}")
        
        print("\n🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        print(f"\n❌ Demo failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
