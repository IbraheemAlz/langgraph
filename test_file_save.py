#!/usr/bin/env python3
"""
Simple test script to verify file saving works in RunPod environment
"""

import json
import time
import os
from pathlib import Path

def test_file_saving():
    """Test file saving in various locations"""
    print("🧪 Testing file saving in RunPod environment")
    print(f"📁 Current directory: {os.getcwd()}")
    print(f"🏃 Pod ID: {os.environ.get('RUNPOD_POD_ID', 'local')}")
    print(f"📍 Workspace: {os.environ.get('WORKSPACE_PATH', 'unknown')}")
    
    # Test data
    test_data = {
        "test": "file saving test",
        "timestamp": time.time(),
        "environment": {
            "cwd": os.getcwd(),
            "pod_id": os.environ.get('RUNPOD_POD_ID', 'local'),
            "workspace": os.environ.get('WORKSPACE_PATH', 'unknown'),
            "user": os.environ.get('USER', 'unknown')
        },
        "sample_results": [
            {"candidate_id": "test_001", "decision": "select"},
            {"candidate_id": "test_002", "decision": "reject"}
        ]
    }
    
    timestamp = int(time.time())
    test_locations = [
        f"./test_output_{timestamp}.json",
        f"results/test_output_{timestamp}.json",
        f"results/json/test_output_{timestamp}.json", 
        f"/workspace/test_output_{timestamp}.json",
        f"/workspace/langgraph/test_output_{timestamp}.json",
        f"/tmp/test_output_{timestamp}.json"
    ]
    
    successful_locations = []
    failed_locations = []
    
    for location in test_locations:
        try:
            print(f"\n📝 Testing: {location}")
            
            # Create directory if needed
            location_path = Path(location)
            location_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"   📂 Directory: {location_path.parent} (exists: {location_path.parent.exists()})")
            
            # Test write permissions on directory
            test_permission_file = location_path.parent / "permission_test.tmp"
            try:
                test_permission_file.write_text("test")
                test_permission_file.unlink()
                print(f"   ✅ Directory writable")
            except Exception as e:
                print(f"   ❌ Directory not writable: {e}")
                failed_locations.append((location, f"Directory not writable: {e}"))
                continue
            
            # Write the actual test file
            with open(location, 'w', encoding='utf-8') as f:
                json.dump(test_data, f, indent=2, ensure_ascii=False)
            print(f"   ✅ File written")
            
            # Verify file exists and has content
            if location_path.exists():
                file_size = location_path.stat().st_size
                print(f"   📄 File size: {file_size} bytes")
                
                if file_size > 0:
                    # Try to read it back
                    with open(location, 'r') as f:
                        loaded_data = json.load(f)
                    print(f"   ✅ File readable - contains {len(loaded_data)} keys")
                    successful_locations.append(location)
                    
                    # Keep the file for manual verification
                    print(f"   💾 File saved successfully at: {location}")
                else:
                    print(f"   ❌ File is empty")
                    failed_locations.append((location, "File is empty"))
            else:
                print(f"   ❌ File does not exist after write")
                failed_locations.append((location, "File does not exist after write"))
                
        except PermissionError as e:
            print(f"   🚫 Permission error: {e}")
            failed_locations.append((location, f"Permission error: {e}"))
        except OSError as e:
            print(f"   💾 OS error: {e}")
            failed_locations.append((location, f"OS error: {e}"))
        except Exception as e:
            print(f"   ⚠️ Unexpected error: {e}")
            failed_locations.append((location, f"Unexpected error: {e}"))
    
    print(f"\n🎯 SUMMARY:")
    print(f"✅ Successful: {len(successful_locations)}")
    print(f"❌ Failed: {len(failed_locations)}")
    
    if successful_locations:
        print(f"\n✅ Working locations:")
        for location in successful_locations:
            print(f"   • {location}")
            
    if failed_locations:
        print(f"\n❌ Failed locations:")
        for location, error in failed_locations:
            print(f"   • {location}: {error}")
    
    # Recommend the best location
    if successful_locations:
        recommended = successful_locations[0]
        print(f"\n💡 Recommended location: {recommended}")
        print(f"Use this path in your batch processor:")
        print(f"python runpod_batch_processor.py --input your_file.csv --output {recommended}")
    else:
        print(f"\n❌ No working locations found! Check permissions and disk space.")
    
    return len(successful_locations) > 0

if __name__ == "__main__":
    success = test_file_saving()
    exit(0 if success else 1) 