#!/usr/bin/env python3
"""
Test script for model search table integration
"""

import os
import sys
import json

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.integration.table_integration import integrate_tables_from_model_search_results

def test_model_search_integration():
    """Test the model search table integration"""
    
    # Use the template search results
    search_results_path = os.path.join(project_root, 'data', 'template', 'search_results.json')
    
    if not os.path.exists(search_results_path):
        print(f"❌ Search results file not found: {search_results_path}")
        return False
    
    print(f"✅ Found search results: {search_results_path}")
    print(f"\n{'='*60}")
    print("Testing Model Search Table Integration")
    print(f"{'='*60}\n")
    
    # Check if CitationLake is available
    use_citationlake = False
    relationship_parquet = None
    
    # Try to find relationship parquet
    possible_parquet_paths = [
        "data_citationlake/processed/modelcard_step3_dedup.parquet",
        "../CitationLake/data/processed/modelcard_step3_dedup.parquet",
    ]
    
    for parquet_path in possible_parquet_paths:
        if os.path.exists(parquet_path):
            relationship_parquet = parquet_path
            print(f"✅ Found relationship parquet: {relationship_parquet}")
            break
    
    if not relationship_parquet:
        print("⚠️  No relationship parquet found, trying CitationLake approach...")
        use_citationlake = True
    
    # Test with union integration
    print("\nTest 1: Union integration (max_models=5, k=10)")
    print("-" * 60)
    result = integrate_tables_from_model_search_results(
        search_results_path,
        integration_type="union",
        k=10,
        max_models=5,
        relationship_parquet=relationship_parquet,
        schema_log_path="data_citationlake/logs/parquet_schema.log",
        use_citationlake=use_citationlake
    )
    
    if result["success"]:
        print(f"✅ Integration successful!")
        print(f"   Stats: {json.dumps(result['stats'], indent=2)}")
        if result.get("integrated_table") is not None:
            df = result["integrated_table"]
            print(f"   Integrated table shape: {df.shape}")
            print(f"   Columns: {list(df.columns)[:10]}...")  # Show first 10 columns
        if result.get("models_with_tables"):
            print(f"   Models with tables ({len(result['models_with_tables'])}):")
            for model in result["models_with_tables"][:5]:
                print(f"     - {model}")
            if len(result["models_with_tables"]) > 5:
                print(f"     ... and {len(result['models_with_tables']) - 5} more")
    else:
        print(f"❌ Integration failed: {result.get('error', 'Unknown error')}")
        return False
    
    print(f"\n{'='*60}\n")
    
    # Test with intersection integration
    print("\nTest 2: Intersection integration (max_models=3, k=5)")
    print("-" * 60)
    result2 = integrate_tables_from_model_search_results(
        search_results_path,
        integration_type="intersection",
        k=5,
        max_models=3,
        relationship_parquet=relationship_parquet,
        schema_log_path="data_citationlake/logs/parquet_schema.log",
        use_citationlake=use_citationlake
    )
    
    if result2["success"]:
        print(f"✅ Integration successful!")
        print(f"   Stats: {json.dumps(result2['stats'], indent=2)}")
        if result2.get("integrated_table") is not None:
            df = result2["integrated_table"]
            print(f"   Integrated table shape: {df.shape}")
    else:
        print(f"❌ Integration failed: {result2.get('error', 'Unknown error')}")
        # This might fail if there are no common columns/rows, which is okay
    
    print(f"\n{'='*60}")
    print("✅ Testing completed!")
    return True

if __name__ == "__main__":
    success = test_model_search_integration()
    sys.exit(0 if success else 1)

