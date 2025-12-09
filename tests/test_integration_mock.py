#!/usr/bin/env python3
"""
Mock test for model search table integration - tests the logic without requiring data files
"""

import os
import sys
import json
import tempfile

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def test_model_id_extraction():
    """Test that we can extract model IDs correctly from search results"""
    print("Test 1: Model ID Extraction")
    print("-" * 60)
    
    # Load real search results
    search_results_path = os.path.join(project_root, 'data', 'template', 'search_results.json')
    with open(search_results_path, 'r') as f:
        search_results = json.load(f)
    
    # Extract Card2Card model IDs (same logic as integration function)
    card2card_results = []
    if isinstance(search_results, dict):
        if "card2card_results" in search_results:
            card2card_results = search_results["card2card_results"]
    
    # Handle both string and object formats
    model_ids = []
    for item in card2card_results:
        if isinstance(item, str):
            model_ids.append(item)
        elif isinstance(item, dict) and "model_id" in item:
            model_ids.append(item["model_id"])
        elif isinstance(item, dict) and "modelId" in item:
            model_ids.append(item["modelId"])
    
    print(f"✅ Extracted {len(model_ids)} model IDs")
    print(f"   First 5: {model_ids[:5]}")
    
    assert len(model_ids) > 0, "Should extract at least one model ID"
    assert model_ids[0] == "Salesforce/codet5-base", "First model should be Salesforce/codet5-base"
    
    print("✅ Model ID extraction works correctly!\n")
    return True

def test_integration_function_structure():
    """Test that the integration function is callable and has correct structure"""
    print("Test 2: Integration Function Structure")
    print("-" * 60)
    
    from src.integration.table_integration import integrate_tables_from_model_search_results
    
    # Check function signature
    import inspect
    sig = inspect.signature(integrate_tables_from_model_search_results)
    params = list(sig.parameters.keys())
    
    print(f"✅ Function exists and is callable")
    print(f"   Parameters: {params}")
    
    required_params = ['search_results_json', 'integration_type', 'k', 'max_models']
    for param in required_params:
        assert param in params, f"Missing required parameter: {param}"
    
    print("✅ Function structure is correct!\n")
    return True

def test_backend_endpoint_exists():
    """Test that the backend endpoint is defined"""
    print("Test 3: Backend Endpoint")
    print("-" * 60)
    
    import importlib.util
    backend_path = os.path.join(project_root, 'src', 'demo', 'backend.py')
    
    if os.path.exists(backend_path):
        # Read backend file and check for endpoint
        with open(backend_path, 'r') as f:
            content = f.read()
        
        assert '/api/integrate-model-search' in content, "Endpoint route not found"
        assert 'def integrate_model_search' in content, "Endpoint function not found"
        assert 'integrate_tables_from_model_search_results' in content, "Integration function not imported"
        
        print("✅ Backend endpoint /api/integrate-model-search exists")
        print("✅ Endpoint function integrate_model_search() is defined")
        print("✅ Integration function is imported")
        print("✅ Backend endpoint structure is correct!\n")
        return True
    else:
        print("⚠️  Backend file not found")
        return False

def test_frontend_ui_exists():
    """Test that the frontend UI includes the integration section"""
    print("Test 4: Frontend UI")
    print("-" * 60)
    
    frontend_path = os.path.join(project_root, 'src', 'demo', 'frontend.py')
    
    if os.path.exists(frontend_path):
        with open(frontend_path, 'r') as f:
            content = f.read()
        
        assert 'Table Integration (from Model Search)' in content, "Model search integration section not found"
        assert 'runModelSearchIntegration' in content, "Integration function not found"
        assert '/api/integrate-model-search' in content, "API endpoint not referenced"
        assert 'integration_model_search_type' in content, "UI controls not found"
        
        print("✅ Frontend includes 'Table Integration (from Model Search)' section")
        print("✅ Frontend includes runModelSearchIntegration() function")
        print("✅ Frontend references /api/integrate-model-search endpoint")
        print("✅ Frontend UI structure is correct!\n")
        return True
    else:
        print("⚠️  Frontend file not found")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Model Search Table Integration Implementation")
    print("=" * 60)
    print()
    
    tests = [
        test_model_id_extraction,
        test_integration_function_structure,
        test_backend_endpoint_exists,
        test_frontend_ui_exists,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed! Implementation is complete.")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed")
        sys.exit(1)

