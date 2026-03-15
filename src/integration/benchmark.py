"""
Benchmark Query Set for Table Integration Testing

Creates benchmark query sets similar to PapersWithCode benchmarks for testing table integration.
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime


def create_benchmark_query_set(
    topic: str,
    query_tables: List[str],
    output_dir: str = "data/benchmarks"
) -> Dict[str, Any]:
    """
    Create a benchmark query set for a specific topic.
    
    Args:
        topic: Topic name (e.g., "transformer_models", "nlp_datasets")
        query_tables: List of table file paths to use as queries
        output_dir: Directory to save benchmark files
        
    Returns:
        Dictionary with benchmark metadata
    """
    os.makedirs(output_dir, exist_ok=True)
    
    benchmark_id = f"{topic}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    benchmark_dir = os.path.join(output_dir, benchmark_id)
    os.makedirs(benchmark_dir, exist_ok=True)
    
    # Create queries directory
    queries_dir = os.path.join(benchmark_dir, "queries")
    os.makedirs(queries_dir, exist_ok=True)
    
    # Copy or reference query tables
    query_info = []
    for i, table_path in enumerate(query_tables):
        if os.path.exists(table_path):
            # Load table to get metadata
            try:
                df = pd.read_csv(table_path, nrows=5)  # Just read first 5 rows for metadata
                query_info.append({
                    "query_id": i,
                    "table_path": table_path,
                    "table_basename": os.path.basename(table_path),
                    "columns": list(df.columns),
                    "sample_rows": len(df)
                })
            except Exception as e:
                print(f"⚠️  Error loading {table_path}: {e}")
    
    # Create benchmark metadata
    benchmark_metadata = {
        "benchmark_id": benchmark_id,
        "topic": topic,
        "created_at": datetime.now().isoformat(),
        "num_queries": len(query_info),
        "queries": query_info,
        "description": f"Benchmark query set for {topic} table integration testing"
    }
    
    # Save metadata
    metadata_path = os.path.join(benchmark_dir, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(benchmark_metadata, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Created benchmark: {benchmark_id}")
    print(f"   Topic: {topic}")
    print(f"   Queries: {len(query_info)}")
    print(f"   Directory: {benchmark_dir}")
    
    return benchmark_metadata


def load_benchmark(benchmark_id: str, benchmark_dir: str = "data/benchmarks") -> Dict[str, Any]:
    """
    Load a benchmark by ID.
    
    Args:
        benchmark_id: Benchmark ID
        benchmark_dir: Directory containing benchmarks
        
    Returns:
        Benchmark metadata dictionary
    """
    benchmark_path = os.path.join(benchmark_dir, benchmark_id, "metadata.json")
    
    if not os.path.exists(benchmark_path):
        raise FileNotFoundError(f"Benchmark not found: {benchmark_id}")
    
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def list_benchmarks(benchmark_dir: str = "data/benchmarks") -> List[Dict[str, Any]]:
    """
    List all available benchmarks.
    
    Args:
        benchmark_dir: Directory containing benchmarks
        
    Returns:
        List of benchmark metadata dictionaries
    """
    if not os.path.exists(benchmark_dir):
        return []
    
    benchmarks = []
    for item in os.listdir(benchmark_dir):
        item_path = os.path.join(benchmark_dir, item)
        if os.path.isdir(item_path):
            metadata_path = os.path.join(item_path, "metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        benchmarks.append(json.load(f))
                except Exception as e:
                    print(f"⚠️  Error loading benchmark {item}: {e}")
    
    return benchmarks


def run_integration_benchmark(
    benchmark_id: str,
    search_results_dir: str = "data",
    integration_type: str = "union",
    k: int = 10
) -> Dict[str, Any]:
    """
    Run integration benchmark on a set of search results.
    
    Args:
        benchmark_id: Benchmark ID to run
        search_results_dir: Directory containing search result JSON files
        integration_type: "union" or "intersection"
        k: Number of tables to integrate
        
    Returns:
        Dictionary with benchmark results
    """
    from .table_integration import integrate_tables_from_search_results
    
    # Load benchmark
    benchmark = load_benchmark(benchmark_id)
    
    # Find search results for each query
    results = []
    
    for query_info in benchmark["queries"]:
        query_id = query_info["query_id"]
        table_basename = query_info["table_basename"]
        
        # Try to find matching search results
        # Look for files that might contain results for this table
        search_result_files = [f for f in os.listdir(search_results_dir) 
                              if f.startswith('compare_search_') and f.endswith('.json')]
        
        # For now, use the most recent search result
        # In a full implementation, we'd match query tables to search results
        if search_result_files:
            latest_result = sorted(search_result_files)[-1]
            search_result_path = os.path.join(search_results_dir, latest_result)
            
            try:
                integration_result = integrate_tables_from_search_results(
                    search_result_path,
                    search_type="single_column",
                    integration_type=integration_type,
                    k=k
                )
                
                if integration_result["success"]:
                    results.append({
                        "query_id": query_id,
                        "table_basename": table_basename,
                        "success": True,
                        "stats": integration_result["stats"]
                    })
                else:
                    results.append({
                        "query_id": query_id,
                        "table_basename": table_basename,
                        "success": False,
                        "error": integration_result.get("error", "Unknown error")
                    })
            except Exception as e:
                results.append({
                    "query_id": query_id,
                    "table_basename": table_basename,
                    "success": False,
                    "error": str(e)
                })
    
    # Summary
    successful = sum(1 for r in results if r.get("success", False))
    total = len(results)
    
    benchmark_results = {
        "benchmark_id": benchmark_id,
        "topic": benchmark["topic"],
        "integration_type": integration_type,
        "k": k,
        "total_queries": total,
        "successful": successful,
        "failed": total - successful,
        "results": results,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save results
    benchmark_dir = os.path.join("data/benchmarks", benchmark_id)
    results_path = os.path.join(benchmark_dir, f"integration_results_{integration_type}_k{k}.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(benchmark_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Benchmark completed: {benchmark_id}")
    print(f"   Successful: {successful}/{total}")
    print(f"   Results saved to: {results_path}")
    
    return benchmark_results


if __name__ == "__main__":
    # Example: Create a simple benchmark
    print("Creating sample benchmark...")
    
    # Find some sample tables (from config)
    from src.config import DEDUPED_HUGGING_CSVS, DEDUPED_GITHUB_CSVS
    sample_dirs = [DEDUPED_HUGGING_CSVS, DEDUPED_GITHUB_CSVS]
    
    sample_tables = []
    for dir_path in sample_dirs:
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
            for f in files[:3]:  # Take first 3
                sample_tables.append(os.path.join(dir_path, f))
            if len(sample_tables) >= 3:
                break
    
    if sample_tables:
        benchmark = create_benchmark_query_set(
            topic="sample_integration_test",
            query_tables=sample_tables
        )
        print(f"\n✅ Created benchmark: {benchmark['benchmark_id']}")
    else:
        print("⚠️  No sample tables found")

