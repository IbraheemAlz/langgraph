import json
import re
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from ..config import Config, PROMPTS
from ..rate_limiter import rate_limited
import time
import re
import json

load_dotenv()

class BiasClassificationAgent:
    """
    Bias Classification Agent for the Multi-Agent AI Hiring System.
    
    This agent acts as an independent fairness auditor, evaluating whether
    hiring decisions were influenced by non-merit factors.
    """
    
    def __init__(self):
        """Initialize the Bias Classification Agent with configured model and prompts."""
        if not Config.validate_environment():
            raise ValueError("Missing required environment variables")
            
        model_config = Config.get_model_config()
        self.llm = ChatGoogleGenerativeAI(**model_config)
        
        self.prompt_template = ChatPromptTemplate.from_template(
            PROMPTS["bias_classification"]
        )
        
        self.feedback_prompt_template = ChatPromptTemplate.from_template(
            PROMPTS["bias_classification_feedback"]
        )

    @rate_limited
    def _invoke_llm_chain(self, chain, params):
        """Rate-limited LLM chain invocation."""
        return chain.invoke(params)

    def _extract_retry_delay_from_error(self, error_message: str) -> int:
        """Extract retry delay from Google API error message."""
        try:
            # Look for retry_delay seconds in the error message
            match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', str(error_message))
            if match:
                return int(match.group(1))
            
            # Fallback: look for other delay patterns
            match = re.search(r'wait\s+(\d+)\s+seconds?', str(error_message), re.IGNORECASE)
            if match:
                return int(match.group(1))
                
        except Exception as e:
            print(f"Could not extract retry delay from error: {e}")
        
        return None

    def _smart_retry_llm_call(self, chain, params):
        """Smart retry function that respects Google's suggested delays."""
        max_retries = 3
        default_delay = 20
        
        for attempt in range(max_retries):
            try:
                return self._invoke_llm_chain(chain, params)
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    raise e
                
                # Extract suggested delay from Google's error message
                suggested_delay = self._extract_retry_delay_from_error(str(e))
                delay = suggested_delay if suggested_delay is not None else default_delay
                
                # Add a small buffer to the suggested delay
                actual_delay = delay + 5 if suggested_delay else delay
                
                print(f"⚠️ Bias Classification attempt {attempt + 1} failed: {str(e)[:200]}...")
                if suggested_delay:
                    print(f"🕒 Google suggests waiting {suggested_delay}s, using {actual_delay}s")
                else:
                    print(f"🕒 Using default delay of {actual_delay}s")
                
                print(f"🔁 Retrying in {actual_delay} seconds...")
                time.sleep(actual_delay)
        
        return None

    def run(self, Resume: str, Job_Description: str, Transcript: str, decision: str, Role: str = "", 
            primary_reason: str = "", original_decision: str = "", previous_feedback: str = "") -> dict:
        """
        Classify whether a hiring decision was biased or unbiased.
        
        Args:
            Resume: Candidate's resume text
            Job_Description: Position requirements
            Transcript: Interview conversation text
            decision: Decision made by job matching agent ("select" or "reject")
            Role: Optional role information
            primary_reason: The main reason provided by job matching agent
            original_decision: For re-evaluations, the original decision
            previous_feedback: For re-evaluations, the previous feedback given
            
        Returns:
            dict: Contains classification and optionally specific_feedback
        """
        try:
            # Determine if this is initial classification or re-evaluation
            is_re_evaluation = bool(original_decision and previous_feedback)
            
            if is_re_evaluation:
                chain = self.feedback_prompt_template | self.llm
                params = {
                    "Resume": Resume,
                    "Job_Description": Job_Description,
                    "Transcript": Transcript,
                    "Role": Role or "Not specified",
                    "decision": decision,
                    "primary_reason": primary_reason,
                    "original_decision": original_decision,
                    "previous_feedback": previous_feedback
                }
            else:
                chain = self.prompt_template | self.llm
                params = {
                    "Resume": Resume,
                    "Job_Description": Job_Description,
                    "Transcript": Transcript,
                    "Role": Role or "Not specified",
                    "decision": decision,
                    "primary_reason": primary_reason
                }
            
            response = self._smart_retry_llm_call(chain, params)
            
            # Log the response for debugging (only if bias is detected)
            result_preview = self._parse_bias_response(response.content)
            if result_preview.get("classification") == "biased":
                evaluation_type = "RE-EVALUATION" if is_re_evaluation else "INITIAL"
                print(f"🔍 BIAS AGENT {evaluation_type} REASONING:")
                print("-" * 50)
                print(response.content)
                print("-" * 50)
            
            return result_preview
                
        except Exception as e:
            print(f"❌ Error in bias classification after all retries: {str(e)}")
            # Default to unbiased in case of error to avoid false positives
            return {
                "classification": Config.DEFAULT_BIAS_ON_ERROR,
                "specific_feedback": None
            }
    
    def _parse_bias_response(self, response_text: str) -> dict:
        """Parse the bias agent response to extract classification and justification from JSON format."""
        
        response_text = response_text.strip()
        result = {
            "classification": "unbiased",
            "specific_feedback": None
        }
        
        try:
            # First try to parse as direct JSON
            parsed = json.loads(response_text)
            
            # Extract classification
            if "classification" in parsed:
                classification = parsed["classification"].lower().strip()
                if classification in ["biased", "unbiased"]:
                    result["classification"] = classification
            
            # Extract justification as feedback
            if "justification" in parsed:
                justification = parsed["justification"]
                if isinstance(justification, str) and justification.strip():
                    result["specific_feedback"] = justification.strip()
                    
            return result
            
        except json.JSONDecodeError:
            # Fallback: Extract JSON from text that might contain extra content
            json_match = re.search(r'\{[^{}]*"classification"[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    
                    if "classification" in parsed:
                        classification = parsed["classification"].lower().strip()
                        if classification in ["biased", "unbiased"]:
                            result["classification"] = classification
                    
                    if "justification" in parsed:
                        justification = parsed["justification"]
                        if isinstance(justification, str) and justification.strip():
                            result["specific_feedback"] = justification.strip()
                            
                    return result
                    
                except json.JSONDecodeError:
                    pass
            
            # Final fallback: Parse old format if JSON parsing fails
            lines = response_text.split('\n')
            for line in lines:
                line = line.strip()
                
                # Extract classification - handle various formats
                if (line.lower().startswith('classification:') or 
                    line.lower().startswith('re-classification:') or
                    line.lower().startswith('**classification:**')):
                    classification_text = line.split(':', 1)[1].strip().lower()
                    # Remove asterisks if present
                    classification_text = classification_text.replace('*', '').strip()
                    if "biased" in classification_text and "unbiased" not in classification_text:
                        result["classification"] = "biased"
                    elif "unbiased" in classification_text:
                        result["classification"] = "unbiased"
                
                # Extract specific feedback - handle various formats
                elif (line.lower().startswith('specific-feedback:') or 
                      line.lower().startswith('additional-feedback:') or
                      line.lower().startswith('**specific-feedback:**') or
                      line.lower().startswith('**additional-feedback:**')):
                    feedback_text = line.split(':', 1)[1].strip()
                    # Remove asterisks if present
                    feedback_text = feedback_text.replace('*', '').strip()
                    if feedback_text and feedback_text.lower() not in ['none', 'n/a', '-']:
                        result["specific_feedback"] = feedback_text
            
            # Final fallback classification check
            if result["classification"] == "unbiased":
                classification_lower = response_text.lower()
                if "biased" in classification_lower and "unbiased" not in classification_lower:
                    result["classification"] = "biased"
        
        return result
