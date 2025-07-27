import json
import re
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from ..config import Config, PROMPTS
from ..rate_limiter import rate_limited

load_dotenv()

class JobMatchingAgent:
    """
    Job Matching Agent for the Multi-Agent AI Hiring System.
    
    This agent acts as the primary hiring decision-maker, evaluating candidates
    based solely on merit-based features like skills, experience, and job alignment.
    """
    
    def __init__(self):
        """Initialize the Job Matching Agent with configured model and prompts."""
        if not Config.validate_environment():
            raise ValueError("Missing required environment variables")
            
        model_config = Config.get_model_config()
        self.llm = ChatGoogleGenerativeAI(**model_config)
        
        self.initial_prompt_template = ChatPromptTemplate.from_template(
            PROMPTS["job_matching_initial"]
        )
        self.feedback_prompt_template = ChatPromptTemplate.from_template(
            PROMPTS["job_matching_feedback"]
        )

    @rate_limited
    def _invoke_llm_chain(self, chain, params):
        """Rate-limited LLM chain invocation."""
        return chain.invoke(params)

    def run(self, Resume: str, Job_Description: str, Transcript: str, Role: str, feedback: str = None) -> dict:
        """
        Make a hiring decision based on candidate information.
        
        Args:
            Resume: Candidate's resume text
            Job_Description: Position requirements
            Transcript: Interview conversation text
            Role: Position title/role
            feedback: Optional feedback for re-evaluation
            
        Returns:
            dict: Contains decision and primary_reason
        """
        try:
            if feedback:
                chain = self.feedback_prompt_template | self.llm
                response = self._invoke_llm_chain(chain, {
                    "Resume": Resume,
                    "Job_Description": Job_Description,
                    "Transcript": Transcript,
                    "Role": Role,
                    "feedback": feedback
                })
            else:
                chain = self.initial_prompt_template | self.llm
                response = self._invoke_llm_chain(chain, {
                    "Resume": Resume,
                    "Job_Description": Job_Description,
                    "Transcript": Transcript,
                    "Role": Role
                })
            
            # Log the response for debugging
            print("🔍 AGENT REASONING:")
            print("-" * 50)
            print(response.content)
            print("-" * 50)
            
            return self._parse_job_matching_response(response.content)
                
        except Exception as e:
            print(f"❌ Error in job matching: {str(e)}")
            # Default to reject in case of error for safety
            return {
                "decision": Config.DEFAULT_DECISION_ON_ERROR,
                "primary_reason": "Error in evaluation process"
            }
    
    def _parse_job_matching_response(self, response_text: str) -> dict:
        """Parse the job matching agent response to extract decision and reasoning from JSON format."""
        
        response_text = response_text.strip()
        result = {
            "decision": "reject",
            "primary_reason": "Could not determine reason"
        }
        
        try:
            # First try to parse as direct JSON
            parsed = json.loads(response_text)
            
            # Extract decision
            if "decision" in parsed:
                decision = parsed["decision"].lower().strip()
                if decision in ["select", "reject"]:
                    result["decision"] = decision
            
            # Extract reasoning - combine array elements or use single string
            if "reasoning" in parsed:
                reasoning = parsed["reasoning"]
                if isinstance(reasoning, list):
                    result["primary_reason"] = " | ".join(reasoning)
                else:
                    result["primary_reason"] = str(reasoning)
                    
            return result
            
        except json.JSONDecodeError:
            # Fallback: Extract JSON from text that might contain extra content
            json_match = re.search(r'\{[^{}]*"decision"[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    
                    if "decision" in parsed:
                        decision = parsed["decision"].lower().strip()
                        if decision in ["select", "reject"]:
                            result["decision"] = decision
                    
                    if "reasoning" in parsed:
                        reasoning = parsed["reasoning"]
                        if isinstance(reasoning, list):
                            result["primary_reason"] = " | ".join(reasoning)
                        else:
                            result["primary_reason"] = str(reasoning)
                            
                    return result
                    
                except json.JSONDecodeError:
                    pass
            
            # Final fallback: Parse old format if JSON parsing fails
            lines = response_text.split('\n')
            for line in lines:
                line = line.strip()
                
                # Extract decision
                if line.lower().startswith('decision:'):
                    decision_text = line.split(':', 1)[1].strip().lower()
                    if "select" in decision_text and "reject" not in decision_text:
                        result["decision"] = "select"
                    elif "reject" in decision_text:
                        result["decision"] = "reject"
                
                # Extract primary reason
                elif line.lower().startswith('primary-reason:') or line.lower().startswith('**primary-reason:**'):
                    if '**' in line:
                        reason_text = line.split('**', 2)[2].strip() if line.count('**') >= 2 else line.split(':', 1)[1].strip()
                    else:
                        reason_text = line.split(':', 1)[1].strip()
                    if reason_text:
                        result["primary_reason"] = reason_text
            
            # Final fallback decision check
            if result["decision"] == "reject":
                decision_lower = response_text.lower()
                if "select" in decision_lower and "reject" not in decision_lower:
                    result["decision"] = "select"
        
        return result
