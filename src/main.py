from typing import TypedDict, List
import logging
from datetime import datetime

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import InMemorySaver

from .agents.job_matching_agent import JobMatchingAgent
from .agents.bias_classification_agent import BiasClassificationAgent
from .config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Define the State Schema - Simplified
class HiringState(TypedDict):
    """State schema for the hiring process."""
    # Input data
    Resume: str
    Job_Description: str
    Transcript: str
    Role: str
    
    # Process state
    decision: str
    primary_reason: str
    bias_classification: str
    re_evaluation_count: int
    bias_feedback: str
    
    # Tracking and insights
    evaluation_insights: List[dict]
    
    # Control
    timestamp: str
    process_complete: bool

# 2. Define Nodes - Simplified
def job_matching_node(state: HiringState) -> dict:
    """Node for making hiring decisions."""
    logger.info("---Job Matching Agent---")
    
    try:
        agent = JobMatchingAgent()
        
        # Check for previous feedback
        re_evaluation_count = state.get("re_evaluation_count", 0)
        bias_feedback = state.get("bias_feedback", "")
        
        if re_evaluation_count > 0 and bias_feedback:
            logger.info(f"🔄 Re-evaluation with bias feedback: {bias_feedback}")
        elif re_evaluation_count > 0:
            logger.info(f"� Re-evaluation #{re_evaluation_count} (no specific feedback)")
        else:
            logger.info("🔄 Initial evaluation")
        
        result = agent.run(
            Resume=state.get('Resume', ''),
            Job_Description=state.get('Job_Description', ''),
            Transcript=state.get('Transcript', ''),
            Role=state.get('Role', ''),
            feedback=bias_feedback if re_evaluation_count > 0 else None
        )
        
        # Extract decision and primary reason
        final_decision = result.get("decision", "reject")
        primary_reason = result.get("primary_reason", "No reason provided")
        
        logger.info(f"Decision: {final_decision}")
        logger.info(f"Primary Reason: {primary_reason}")
        
        # Track evaluation insight
        evaluation_number = re_evaluation_count + 1
        evaluation_insight = {
            "evaluation_number": evaluation_number,
            "decision": final_decision,
            "primary_reason": primary_reason,
            "agent": "job_matching",
            "is_re_evaluation": re_evaluation_count > 0
        }
        
        # Initialize or update evaluation insights list
        evaluation_insights = state.get("evaluation_insights", [])
        evaluation_insights.append(evaluation_insight)
        
        return {
            "decision": final_decision,
            "primary_reason": primary_reason,
            "evaluation_insights": evaluation_insights
        }
        
    except Exception as e:
        logger.error(f"Error in job matching: {str(e)}")
        evaluation_insights = state.get("evaluation_insights", [])
        evaluation_insights.append({
            "evaluation_number": state.get("re_evaluation_count", 0) + 1,
            "decision": "reject",
            "primary_reason": "Error in evaluation process",
            "agent": "job_matching",
            "is_re_evaluation": state.get("re_evaluation_count", 0) > 0,
            "error": str(e)
        })
        return {
            "decision": "reject",
            "primary_reason": "Error in evaluation process",
            "evaluation_insights": evaluation_insights
        }

def bias_classification_node(state: HiringState) -> dict:
    """Node for bias classification."""
    logger.info("---Bias Classification Agent---")
    
    try:
        agent = BiasClassificationAgent()
        
        # Get re-evaluation context
        re_evaluation_count = state.get("re_evaluation_count", 0)
        original_decision = None
        previous_feedback = None
        
        # For re-evaluations, get original decision and previous feedback
        if re_evaluation_count > 0:
            evaluation_insights = state.get("evaluation_insights", [])
            if len(evaluation_insights) >= 2:  # Should have original and current
                original_decision = evaluation_insights[0].get("decision", "")
                # Get previous bias feedback if available
                for insight in evaluation_insights:
                    if insight.get("classification") == "biased" and insight.get("specific_feedback"):
                        previous_feedback = insight.get("specific_feedback")
                        break
        
        result = agent.run(
            Resume=state.get('Resume', ''),
            Job_Description=state.get('Job_Description', ''),
            Transcript=state.get('Transcript', ''),
            decision=state.get('decision', ''),
            Role=state.get('Role', ''),
            primary_reason=state.get('primary_reason', ''),
            original_decision=original_decision or "",
            previous_feedback=previous_feedback or ""
        )
        
        # Extract classification and feedback
        bias_classification = result.get("classification", "unbiased")
        specific_feedback = result.get("specific_feedback", None)
        
        # Only log when bias is detected
        if bias_classification == "biased":
            logger.info(f"Bias Classification: {bias_classification}")
            if specific_feedback:
                logger.info(f"Bias Feedback: {specific_feedback}")
        
        # Update the most recent evaluation insight with bias classification
        evaluation_insights = state.get("evaluation_insights", [])
        if evaluation_insights:
            evaluation_insights[-1]["classification"] = bias_classification
            if specific_feedback:
                evaluation_insights[-1]["specific_feedback"] = specific_feedback
        
        result_dict = {
            "bias_classification": bias_classification,
            "evaluation_insights": evaluation_insights
        }
        
        # Add bias feedback to state if bias is detected
        if bias_classification == "biased" and specific_feedback:
            result_dict["bias_feedback"] = specific_feedback
        
        return result_dict
        
    except Exception as e:
        logger.error(f"Error in bias classification: {str(e)}")
        evaluation_insights = state.get("evaluation_insights", [])
        if evaluation_insights:
            evaluation_insights[-1]["classification"] = "unbiased"
            evaluation_insights[-1]["error"] = str(e)
        
        return {
            "bias_classification": "unbiased",
            "evaluation_insights": evaluation_insights
        }

def should_continue(state: HiringState) -> str:
    """Determine if we should continue or end after bias classification."""
    bias_classification = state.get("bias_classification", "unbiased")
    re_evaluation_count = state.get("re_evaluation_count", 0)
    max_re_evaluations = Config.MAX_RE_EVALUATIONS
    
    if bias_classification == "biased" and re_evaluation_count < max_re_evaluations:
        logger.info(f"---Bias detected, re-evaluating (attempt {re_evaluation_count + 1})---")
        return "re_evaluate"
    
    logger.info("---Decision finalized---")
    return "finalize"

def re_evaluate_node(state: HiringState) -> dict:
    """Increment re-evaluation counter for bias-driven re-evaluation."""
    count = state.get("re_evaluation_count", 0) + 1
    bias_feedback = state.get("bias_feedback", "")
    logger.info(f"Re-evaluation #{count} triggered by bias detection")
    
    # The bias_feedback is already in state from bias classification node
    # We need to ensure it persists through the re-evaluation
    result = {
        "re_evaluation_count": count
    }
    
    # Explicitly preserve bias_feedback if it exists
    if bias_feedback:
        result["bias_feedback"] = bias_feedback
    
    return result

def finalize_node(state: HiringState) -> dict:
    """Finalize the hiring decision."""
    logger.info("---Finalizing Decision---")
    
    decision = state.get("decision", "reject")
    bias_classification = state.get("bias_classification", "unbiased")
    re_evaluations = state.get("re_evaluation_count", 0)
    evaluation_insights = state.get("evaluation_insights", [])
    
    logger.info(f"Final Decision: {decision}")
    logger.info(f"Bias Classification: {bias_classification}")
    logger.info(f"Re-evaluations: {re_evaluations}")
    
    # Log evaluation insights summary
    if evaluation_insights:
        logger.info("📊 Evaluation Insights:")
        for insight in evaluation_insights:
            eval_type = "re-evaluation" if insight.get("is_re_evaluation") else "initial"
            logger.info(f"  {eval_type} #{insight['evaluation_number']}: {insight['decision']} → {insight.get('classification', 'pending')}")
    
    return {
        "process_complete": True,
        "timestamp": datetime.now().isoformat(),
        "evaluation_insights": evaluation_insights
    }

# 3. Build the Graph - Simplified
def create_hiring_workflow():
    """Create and return the hiring workflow graph."""
    
    workflow = StateGraph(HiringState)
    
    # Add nodes
    workflow.add_node("job_matcher", job_matching_node)
    workflow.add_node("bias_classifier", bias_classification_node)
    workflow.add_node("re_evaluate", re_evaluate_node)
    workflow.add_node("finalize", finalize_node)
    
    # Add edges - cleaner flow
    workflow.add_edge(START, "job_matcher")
    workflow.add_edge("job_matcher", "bias_classifier")
    workflow.add_conditional_edges(
        "bias_classifier", 
        should_continue,
        {
            "re_evaluate": "re_evaluate",
            "finalize": "finalize"
        }
    )
    workflow.add_edge("re_evaluate", "job_matcher")
    workflow.add_edge("finalize", END)
    
    # Add memory
    memory = InMemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app
