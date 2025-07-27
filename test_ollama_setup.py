#!/usr/bin/env python3
"""
Test script to verify the RunPod Ollama setup is working correctly.
Run this after deploying to RunPod to ensure all components are functional.
"""

import sys
import os
import requests
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_ollama_connection():
    """Test basic Ollama connection"""
    print("🔍 Testing Ollama connection...")
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=10)
        if response.status_code == 200:
            version_info = response.json()
            print(f"✅ Ollama connected successfully - Version: {version_info.get('version', 'unknown')}")
            return True
        else:
            print(f"❌ Ollama returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        return False

def test_model_availability():
    """Test if the Gemma model is available"""
    print("🔍 Testing model availability...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json()
            available_models = [model['name'] for model in models.get('models', [])]
            print(f"📋 Available models: {available_models}")
            
            if 'gemma3:27b-instruct' in available_models:
                print("✅ Gemma 3 27B Instruct model is available")
                return True
            else:
                print("⚠️ Gemma 3 27B Instruct model not found. Available models:", available_models)
                return False
        else:
            print(f"❌ Failed to get model list: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model check failed: {e}")
        return False

def test_model_inference():
    """Test model inference with a simple prompt"""
    print("🔍 Testing model inference...")
    try:
        payload = {
            "model": "gemma3:27b-instruct",
            "prompt": "Hello, please respond with 'AI system ready' if you can understand this message.",
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_ctx": 4096
            }
        }
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            ai_response = result.get('response', '')
            print(f"✅ Model inference successful")
            print(f"📤 AI Response: {ai_response.strip()}")
            return True
        else:
            print(f"❌ Model inference failed: {response.status_code}")
            print(f"📄 Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Model inference error: {e}")
        return False

def test_agents():
    """Test both agents"""
    print("🔍 Testing agents...")
    try:
        from src.config import Config
        from src.agents.job_matching_agent import JobMatchingAgent
        from src.agents.bias_classification_agent import BiasClassificationAgent
        
        # Test configuration
        config = Config()
        print(f"✅ Configuration loaded - Model: {config.MODEL_NAME}")
        
        # Test job matching agent
        job_agent = JobMatchingAgent()
        print("✅ Job Matching Agent initialized")
        
        # Test bias classification agent
        bias_agent = BiasClassificationAgent()
        print("✅ Bias Classification Agent initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 RunPod Ollama Setup Test")
    print("=" * 50)
    
    tests = [
        ("Ollama Connection", test_ollama_connection),
        ("Model Availability", test_model_availability),
        ("Model Inference", test_model_inference),
        ("Agent Initialization", test_agents)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔄 Running {test_name}...")
        result = test_func()
        results.append((test_name, result))
        print()
    
    # Summary
    print("=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! RunPod setup is ready for production.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
