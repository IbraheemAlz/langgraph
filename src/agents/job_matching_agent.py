import json
import re
import requests
import logging
from typing import Dict, Any
from ..config import Config, PROMPTS

class JobMatchingAgent:
    """
    Job Matching Agent for the Multi-Agent AI Hiring System - RunPod Optimized.
    
    This agent acts as the primary hiring decision-maker, evaluating candidates
    using local Ollama model for high-performance processing.
    """
    
    def __init__(self):
        """Initialize the Job Matching Agent with local Ollama."""
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
                "gpu_memory_utilization": 0.95,             # Use 95% GPU memory
                "num_thread": 1,                             # Minimal CPU threads for GPU mode
                "use_mlock": False,                          # Disable to avoid warnings
                "use_mmap": True,                            # Memory mapping for speed
                "numa": False,                               # Disable NUMA for single GPU
                "max_tokens": self.config.MAX_TOKENS,        # 🚀 Add explicit token limit
                "repeat_penalty": 1.0,                      # 🚀 Disable penalty for speed
                "top_k": 20                                  # 🚀 Reduce top_k for faster sampling
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
    def run(self, Resume: str, Job_Description: str, Transcript: str, Role: str, feedback: str = None) -> dict:
        """
        Make a hiring decision based on candidate information using Ollama.
        
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
                prompt = PROMPTS["job_matching_feedback"].format(
                    Resume=Resume,
                    Job_Description=Job_Description,
                    Transcript=Transcript,
                    Role=Role,
                    feedback=feedback
                )
            else:
                prompt = PROMPTS["job_matching_initial"].format(
                    Resume=Resume,
                    Job_Description=Job_Description,
                    Transcript=Transcript,
                    Role=Role
                )
            
            # Get response from Ollama
            response = self._call_ollama(prompt)
            
            # Log response for debugging (only in development mode)
            if self.config.LOG_LEVEL == "DEBUG":
                self.logger.debug("🔍 AGENT REASONING:")
                self.logger.debug("-" * 50)
                self.logger.debug(response)
                self.logger.debug("-" * 50)
            
            return self._parse_job_matching_response(response)
                
        except Exception as e:
            self.logger.error(f"❌ Error in job matching: {str(e)}")
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
