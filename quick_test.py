#!/usr/bin/env python3
"""Quick test to verify all imports work"""

try:
    from src.config import Config
    print("✅ Config import successful")
except Exception as e:
    print(f"❌ Config import failed: {e}")

try:
    from src.agents.job_matching_agent import JobMatchingAgent
    print("✅ Job matching agent import successful")
except Exception as e:
    print(f"❌ Job matching agent import failed: {e}")

try:
    from src.agents.bias_classification_agent import BiasClassificationAgent
    print("✅ Bias classification agent import successful")
except Exception as e:
    print(f"❌ Bias classification agent import failed: {e}")

try:
    import fastapi
    print("✅ FastAPI import successful")
except Exception as e:
    print(f"❌ FastAPI import failed: {e}")

try:
    import uvicorn
    print("✅ Uvicorn import successful")
except Exception as e:
    print(f"❌ Uvicorn import failed: {e}")

try:
    import psutil
    print("✅ psutil import successful")
except Exception as e:
    print(f"❌ psutil import failed: {e}")

print("\n🎉 All import tests completed!")
