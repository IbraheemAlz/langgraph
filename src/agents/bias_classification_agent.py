import json
import re
import requests
import logging
from typing import Dict, Any
from ..config import Config, PROMPTS

class BiasClassificationAgent:
    """
    Bias Classification Agent for the Multi-Agent AI Hiring System - RunPod Optimized.
    
    This agent acts as an independent fairness auditor, evaluating whether
    hiring decisions were influenced by non-merit factors using local Ollama.
    """
    
    def __init__(self):
        """Initialize the Bias Classification Agent with local Ollama."""
        if not Config.validate_environment():
            raise ValueError("Ollama service not available")
            
        self.config = Config()
        self.ollama_url = f"{self.config.OLLAMA_BASE_URL}/api/generate"
        self.model_name = self.config.MODEL_NAME
        self.logger = logging.getLogger(__name__)
        
        # Verify Ollama connection
        self._verify_ollama_connection()
    
    def _verify_ollama_connection(self):
        """Verify Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.config.OLLAMA_BASE_URL}/api/version", timeout=10)
            if response.status_code == 200:
                self.logger.info("✅ Ollama connection verified")
            else:
                raise ConnectionError("Ollama not responding")
        except Exception as e:
            self.logger.error(f"❌ Ollama connection failed: {e}")
            raise
    
    def _call_ollama(self, prompt: str) -> str:
        """Make API call to local Ollama instance with H100 optimizations"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.TEMPERATURE,
                "top_p": self.config.TOP_P,
                "num_ctx": self.config.MODEL_CONTEXT_LENGTH,
                "num_batch": self.config.OLLAMA_BATCH_SIZE,  # H100 batch optimization
                "num_gpu": 99,                               # Force ALL layers to GPU (critical fix!)
                "num_thread": 1,                             # Minimal CPU threads for GPU mode
                "use_mmap": True,                            # Memory mapping for speed
                "numa": False,                               # Valid parameter for NUMA control
                "repeat_penalty": 1.0,                      # Valid parameter for repetition control
                "top_k": 20,                                 # Valid parameter for top-k sampling
                "num_predict": self.config.MAX_TOKENS        # Use num_predict instead of max_tokens
            }
        }
        
        try:
            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=self.config.REQUEST_TIMEOUT
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', '').strip()
            
        except requests.exceptions.Timeout:
            self.logger.error("Request timeout - model may be overloaded")
            raise
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Ollama request failed: {e}")
            raise

    def run(self, Resume: str, Job_Description: str, Transcript: str, decision: str, Role: str = "", 
            primary_reason: str = "", original_decision: str = "", previous_feedback: str = "") -> dict:
        """
        Classify whether a hiring decision was biased or unbiased using Ollama.
        
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
                prompt = PROMPTS["bias_classification_feedback"].format(
                    Resume=Resume,
                    Job_Description=Job_Description,
                    Transcript=Transcript,
                    Role=Role or "Not specified",
                    decision=decision,
                    primary_reason=primary_reason,
                    original_decision=original_decision,
                    previous_feedback=previous_feedback
                )
            else:
                prompt = PROMPTS["bias_classification"].format(
                    Resume=Resume,
                    Job_Description=Job_Description,
                    Transcript=Transcript,
                    Role=Role or "Not specified",
                    decision=decision,
                    primary_reason=primary_reason
                )
            
            # Get response from Ollama
            response = self._call_ollama(prompt)
            
            # Log the response for debugging (only if bias is detected)
            result_preview = self._parse_bias_response(response)
            if result_preview.get("classification") == "biased":
                evaluation_type = "RE-EVALUATION" if is_re_evaluation else "INITIAL"
                print(f"🔍 BIAS AGENT {evaluation_type} REASONING:")
                print("-" * 50)
                print(response)
                print("-" * 50)
            
            return result_preview
                
        except Exception as e:
            self.logger.error(f"❌ Error in bias classification: {str(e)}")
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
            
            # Extract feedback - handle both old and new formats
            if "specific_feedback" in parsed:
                feedback = parsed["specific_feedback"]
                if isinstance(feedback, str) and feedback.strip():
                    result["specific_feedback"] = feedback.strip()
            # Fallback to old format for compatibility
            elif "justification" in parsed:
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
                    
                    # Check for new format first
                    if "specific_feedback" in parsed:
                        feedback = parsed["specific_feedback"]
                        if isinstance(feedback, str) and feedback.strip():
                            result["specific_feedback"] = feedback.strip()
                    # Fallback to old format for compatibility
                    elif "justification" in parsed:
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
