"""
Test script for table integration functionality.

Tests integration with a few sample tables.
"""

import os
import sys
import json

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.config import TABLE_BASE_DIRS
from src.integration.table_integration import integrate_tables, integrate_tables_from_search_results


def test_basic_integration():
    """Test basic integration with sample tables."""
    print("=" * 60)
    print("Test 1: Basic Union Integration")
    print("=" * 60)
    
    # Find some sample tables (dirs from config)
    sample_dirs = list(TABLE_BASE_DIRS)
    
    table_paths = []
    for dir_path in sample_dirs:
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
            for f in files[:3]:  # Take first 3 tables
                table_paths.append(os.path.join(dir_path, f))
            if len(table_paths) >= 3:
                break
    
    if len(table_paths) < 2:
        print("⚠️  Not enough sample tables found. Need at least 2 tables.")
        print(f"   Found: {len(table_paths)} tables")
        return
    
    print(f"Using {len(table_paths)} tables for integration:")
    for path in table_paths:
        print(f"  - {os.path.basename(path)}")
    
    # Test union integration
    result = integrate_tables(table_paths[:3], integration_type="union", k=50)
    
    if result["success"]:
        print(f"\n✅ Integration successful!")
        print(f"   Output shape: {result['integrated_table'].shape}")
        print(f"\nFirst few rows:")
        print(result['integrated_table'].head())
    else:
        print(f"\n❌ Integration failed: {result.get('error', 'Unknown error')}")


def test_intersection_integration():
    """Test intersection integration."""
    print("\n" + "=" * 60)
    print("Test 2: Intersection Integration")
    print("=" * 60)
    
    # Find sample tables (dirs from config)
    sample_dirs = [
        TABLE_BASE_DIRS[0],
        TABLE_BASE_DIRS[1],
    ]
    
    table_paths = []
    for dir_path in sample_dirs:
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
            for f in files[:2]:  # Take first 2 tables
                table_paths.append(os.path.join(dir_path, f))
            if len(table_paths) >= 2:
                break
    
    if len(table_paths) < 2:
        print("⚠️  Not enough sample tables found. Need at least 2 tables.")
        return
    
    print(f"Using {len(table_paths)} tables for intersection:")
    for path in table_paths:
        print(f"  - {os.path.basename(path)}")
    
    # Test intersection integration
    result = integrate_tables(table_paths[:2], integration_type="intersection", k=50)
    
    if result["success"]:
        print(f"\n✅ Integration successful!")
        print(f"   Output shape: {result['integrated_table'].shape}")
        if len(result['integrated_table']) > 0:
            print(f"\nFirst few rows:")
            print(result['integrated_table'].head())
        else:
            print("\n⚠️  No common rows found (empty intersection)")
    else:
        print(f"\n❌ Integration failed: {result.get('error', 'Unknown error')}")


def test_from_search_results():
    """Test integration from search results JSON."""
    print("\n" + "=" * 60)
    print("Test 3: Integration from Search Results")
    print("=" * 60)
    
    # Look for recent search results
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"⚠️  Data directory not found: {data_dir}")
        return
    
    # Find most recent search results
    json_files = [f for f in os.listdir(data_dir) if f.startswith('compare_search_') and f.endswith('.json')]
    
    if not json_files:
        print("⚠️  No search results found. Please run a search first.")
        return
    
    # Use most recent file
    latest_file = sorted(json_files)[-1]
    search_results_path = os.path.join(data_dir, latest_file)
    
    print(f"Using search results: {latest_file}")
    
    # Test integration from search results
    result = integrate_tables_from_search_results(
        search_results_path,
        search_type="single_column",
        integration_type="union",
        k=10
    )
    
    if result["success"]:
        print(f"\n✅ Integration successful!")
        print(f"   Output shape: {result['integrated_table'].shape}")
        print(f"\nFirst few rows:")
        print(result['integrated_table'].head())
    else:
        print(f"\n❌ Integration failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    print("Testing Table Integration Functionality\n")
    
    # Test 1: Basic union integration
    test_basic_integration()
    
    # Test 2: Intersection integration
    test_intersection_integration()
    
    # Test 3: Integration from search results
    test_from_search_results()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)

