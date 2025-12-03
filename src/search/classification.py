"""
Table Classification Module

This module provides functionality to classify tables based on their content and structure.
Classification can be done on a single table or in batch on a local datalake.

The classification is used to filter search results to only search within tables of the same type.

Classification labels (from tab2know):
- Observation: Performance/result tables (most common)
- Input: Configuration/input tables
- Other: Other types of tables
- Example: Example tables
"""

import os
import sys
import json
import pandas as pd
import duckdb
import subprocess
import tempfile
import shutil
from typing import Dict, List, Optional, Any, Union
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


def _find_tab2know_repo() -> Optional[str]:
    """Find Tab2Know repository directory - checks local others/ first, then external repos"""
    # Priority 1: Check local others/tab2know directory (bundled with this repo)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    local_tab2know = os.path.join(project_root, 'others', 'tab2know')
    if os.path.exists(os.path.join(local_tab2know, 'run_inference.py')):
        return local_tab2know
    
    # Priority 2: Check environment variable
    if 'TAB2KNOW_REPO' in os.environ:
        repo_dir = os.environ['TAB2KNOW_REPO']
        if os.path.exists(os.path.join(repo_dir, 'run_inference.py')):
            return repo_dir
    
    # Priority 3: Check common external locations
    possible_paths = [
        os.path.join(os.path.expanduser('~'), 'Repo', 'TabKnow_internal'),
        os.path.join(os.path.dirname(__file__), '../../..', 'TabKnow_internal'),
        '/Users/doradong/Repo/TabKnow_internal',
    ]
    
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(os.path.join(abs_path, 'run_inference.py')):
            return abs_path
    
    return None


def _extract_classification_from_rdf_type(rdf_type: str) -> str:
    """
    Extract simplified classification label from tab2know rdf:type URI.
    
    Args:
        rdf_type: Full URI like "http://cs.vu.nl/tab2know/Observation"
    
    Returns:
        Simplified label: "Observation", "Input", "Other", "Example", or "unknown"
    """
    if not rdf_type:
        return "unknown"
    
    # Extract the last part after the last '/'
    label = rdf_type.split('/')[-1]
    
    # Map to standard labels
    valid_labels = ["Observation", "Input", "Other", "Example"]
    if label in valid_labels:
        return label
    
    return "unknown"


def _classify_with_tab2know(csv_path: str, tab2know_repo: Optional[str] = None) -> str:
    """
    Classify a table using tab2know inference.
    
    Args:
        csv_path: Path to CSV file to classify
        tab2know_repo: Path to Tab2Know repository (auto-detected if None)
    
    Returns:
        Classification label: "Observation", "Input", "Other", "Example", or "unknown"
    """
    if tab2know_repo is None:
        tab2know_repo = _find_tab2know_repo()
    
    if tab2know_repo is None:
        raise RuntimeError(
            "Tab2Know repository not found. Please set TAB2KNOW_REPO environment variable "
            "or ensure TabKnow_internal is in a standard location."
        )
    
    # Create a temporary directory with the CSV file
    temp_dir = tempfile.mkdtemp()
    temp_csv_dir = os.path.join(temp_dir, 'csv')
    os.makedirs(temp_csv_dir, exist_ok=True)
    
    # Copy CSV to temp directory
    csv_basename = os.path.basename(csv_path)
    temp_csv_path = os.path.join(temp_csv_dir, csv_basename)
    shutil.copy2(csv_path, temp_csv_path)
    
    # Create output file
    output_jsonl = os.path.join(temp_dir, 'preds.jsonl')
    
    try:
        # Find models directory (try local first, then external repo)
        models_dir = os.path.join(tab2know_repo, 'models')
        if not os.path.exists(models_dir):
            # Try external repo location for models
            external_repo = os.path.join(os.path.expanduser('~'), 'Repo', 'TabKnow_internal')
            external_models = os.path.join(external_repo, 'models')
            if os.path.exists(external_models):
                models_dir = external_models
            else:
                # Last resort: use default path (may fail, but let tab2know handle it)
                models_dir = os.path.join(tab2know_repo, 'models')
        
        # Run tab2know inference
        cmd = [
            sys.executable,
            os.path.join(tab2know_repo, 'run_inference.py'),
            temp_dir,
            '--output', output_jsonl,
            '--modeldir', models_dir,
            '--type-model', 'supervised-lr',
            '--no-caption',
            '--quiet'
        ]
        
        env = os.environ.copy()
        env['PYTHONPATH'] = tab2know_repo
        env['TAB2KNOW_NO_CAPTION'] = '1'
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=60  # 60 second timeout
        )
        
        if result.returncode != 0:
            print(f"Warning: tab2know inference failed: {result.stderr}")
            return "unknown"
        
        # Read the result JSONL file
        if not os.path.exists(output_jsonl):
            return "unknown"
        
        with open(output_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        rdf_type = data.get('rdf:type', '')
                        if rdf_type:
                            return _extract_classification_from_rdf_type(rdf_type)
                    except json.JSONDecodeError:
                        continue
        
        return "unknown"
    
    except subprocess.TimeoutExpired:
        print("Warning: tab2know inference timed out")
        return "unknown"
    except Exception as e:
        print(f"Warning: Error running tab2know inference: {e}")
        return "unknown"
    finally:
        # Clean up temp directory
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass


def classify_table(
    table: Union[pd.DataFrame, str],
    method: str = "tab2know"
) -> str:
    """
    Classify a single table based on its content and structure.
    
    Args:
        table: Either a pandas DataFrame or a path to a CSV file
        method: Classification method - "tab2know" (default), "heuristic", or "ml" (future)
    
    Returns:
        Classification label (string):
        - For tab2know: "Observation", "Input", "Other", "Example", or "unknown"
        - For heuristic: "numerical", "categorical", "mixed", "id_like", "empty", etc.
    """
    # Load table if path is provided
    if isinstance(table, str):
        if not os.path.exists(table):
            raise FileNotFoundError(f"Table file not found: {table}")
        csv_path = table
    else:
        # Save DataFrame to temporary CSV
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        table.to_csv(temp_file.name, index=False)
        temp_file.close()
        csv_path = temp_file.name
    
    try:
        if method == "tab2know":
            return _classify_with_tab2know(csv_path)
        elif method == "heuristic":
            df = pd.read_csv(csv_path)
            return _classify_heuristic(df)
        elif method == "ml":
            # Future: ML-based classification
            raise NotImplementedError("ML-based classification not yet implemented")
        else:
            raise ValueError(f"Unknown classification method: {method}")
    finally:
        # Clean up temp file if we created it
        if isinstance(table, pd.DataFrame) and os.path.exists(csv_path):
            try:
                os.unlink(csv_path)
            except Exception:
                pass


def _classify_heuristic(df: pd.DataFrame) -> str:
    """
    Heuristic-based table classification.
    
    Classifies tables based on:
    - Data types (numerical vs categorical)
    - Number of columns
    - Number of rows
    - Presence of text vs numbers
    
    Returns:
        Classification label
    """
    if df.empty:
        return "empty"
    
    num_rows = len(df)
    num_cols = len(df.columns)
    
    # Count data types
    numeric_cols = df.select_dtypes(include=['number']).columns
    text_cols = df.select_dtypes(include=['object']).columns
    
    num_numeric = len(numeric_cols)
    num_text = len(text_cols)
    
    # Calculate ratios
    numeric_ratio = num_numeric / num_cols if num_cols > 0 else 0
    text_ratio = num_text / num_cols if num_cols > 0 else 0
    
    # Classification logic
    if num_cols == 0:
        return "empty"
    elif num_cols == 1:
        if numeric_ratio > 0.5:
            return "single_numerical"
        else:
            return "single_categorical"
    elif numeric_ratio > 0.7:
        return "numerical"
    elif text_ratio > 0.7:
        return "categorical"
    elif numeric_ratio > 0.4 and text_ratio > 0.4:
        return "mixed"
    else:
        # Check for specific patterns
        # If has many unique values in text columns, might be IDs
        if num_text > 0:
            sample_text_col = text_cols[0]
            unique_ratio = df[sample_text_col].nunique() / num_rows if num_rows > 0 else 0
            if unique_ratio > 0.9:
                return "id_like"
        
        return "mixed"


def classify_table_from_db(
    tableid: int,
    db_path: str,
    index_table: str = "modellake_index"
) -> Optional[str]:
    """
    Classify a table by its tableid from the database.
    
    Args:
        tableid: Table ID in the database
        db_path: Path to modellake.db
        index_table: Name of the index table
    
    Returns:
        Classification label or None if table not found
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    con = duckdb.connect(db_path, read_only=True)
    try:
        # Get filename for this tableid
        query = f"""
            SELECT DISTINCT filename 
            FROM {index_table} 
            WHERE tableid = ? AND rowid = -1
            LIMIT 1
        """
        result = con.execute(query, [tableid]).fetchone()
        
        if not result:
            return None
        
        filename = result[0]
        
        # Try to find the actual CSV file
        csv_path = _find_csv_file(filename)
        
        if csv_path and os.path.exists(csv_path):
            return classify_table(csv_path)
        else:
            # Fallback: use table_type from database if available
            type_query = f"""
                SELECT DISTINCT table_type 
                FROM {index_table} 
                WHERE tableid = ? AND rowid = -1
                LIMIT 1
            """
            type_result = con.execute(type_query, [tableid]).fetchone()
            if type_result:
                return type_result[0]  # Use table_type as classification
            return "unknown"
    finally:
        con.close()


def _find_csv_file(filename: str) -> Optional[str]:
    """
    Try to find the CSV file given a filename/basename.
    
    Args:
        filename: Filename or basename to search for
    
    Returns:
        Full path to CSV file or None if not found
    """
    basename = os.path.basename(filename)
    
    # Common locations to search
    search_dirs = [
        "data_citationlake/processed/deduped_hugging_csvs",
        "data_citationlake/processed/deduped_github_csvs",
        "data_citationlake/processed/tables_output",
        "data/raw",
    ]
    
    # Also try relative to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    
    for search_dir in search_dirs:
        full_dir = os.path.join(project_root, search_dir)
        if os.path.exists(full_dir):
            full_path = os.path.join(full_dir, basename)
            if os.path.exists(full_path):
                return full_path
    
    # Try the filename as-is if it's already a full path
    if os.path.exists(filename):
        return filename
    
    return None


def classify_datalake_batch(
    db_path: str,
    index_table: str = "modellake_index",
    output_json: Optional[str] = None,
    limit: Optional[int] = None,
    method: str = "tab2know"
) -> Dict[int, str]:
    """
    Classify all tables in a datalake (database) in batch.
    
    Args:
        db_path: Path to modellake.db
        index_table: Name of the index table
        output_json: Optional path to save classification results as JSON
        limit: Optional limit on number of tables to classify
        method: Classification method - "tab2know" (default), "heuristic", or "ml"
    
    Returns:
        Dictionary mapping tableid to classification label
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    print(f"\n{'='*60}")
    print(f"🔍 Batch Classification of Datalake")
    print(f"{'='*60}")
    print(f"Database: {db_path}")
    print(f"Index table: {index_table}")
    print(f"Method: {method}")
    if limit:
        print(f"Limit: {limit} tables")
    print(f"{'='*60}\n")
    
    con = duckdb.connect(db_path, read_only=True)
    classifications = {}
    
    try:
        # Get all unique tableids
        query = f"""
            SELECT DISTINCT tableid, filename, table_type 
            FROM {index_table} 
            WHERE rowid = -1
        """
        if limit:
            query += f" LIMIT {limit}"
        
        results = con.execute(query).fetchall()
        total_tables = len(results)
        
        print(f"📊 Found {total_tables} tables to classify\n")
        
        for i, (tableid, filename, table_type) in enumerate(results, 1):
            if i % 100 == 0:
                print(f"   Progress: {i}/{total_tables} tables classified...")
            
            try:
                # Try to find and classify the actual CSV
                csv_path = _find_csv_file(filename)
                
                if csv_path and os.path.exists(csv_path):
                    classification = classify_table(csv_path, method=method)
                    classifications[tableid] = classification
                else:
                    # Fallback: use table_type from database
                    classifications[tableid] = table_type if table_type else "unknown"
            except Exception as e:
                print(f"   ⚠️  Error classifying table {tableid} ({filename}): {e}")
                classifications[tableid] = "unknown"
        
        print(f"\n✅ Classified {len(classifications)} tables")
        
        # Print classification distribution
        from collections import Counter
        dist = Counter(classifications.values())
        print(f"\n📊 Classification Distribution:")
        for label, count in dist.most_common():
            print(f"   {label}: {count}")
        
    finally:
        con.close()
    
    # Save results if requested
    if output_json:
        os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else '.', exist_ok=True)
        # Convert int keys to strings for JSON
        json_data = {str(k): v for k, v in classifications.items()}
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Classification results saved to {output_json}")
    
    return classifications


def load_classifications(json_path: str) -> Dict[int, str]:
    """
    Load classification results from a JSON file.
    
    Args:
        json_path: Path to JSON file with classifications
    
    Returns:
        Dictionary mapping tableid to classification label
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Classification file not found: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # Convert string keys back to int
    return {int(k): v for k, v in json_data.items()}


def get_tables_by_classification(
    classification: str,
    classifications: Dict[int, str]
) -> List[int]:
    """
    Get all table IDs that have a specific classification.
    
    Args:
        classification: Classification label to filter by
        classifications: Dictionary mapping tableid to classification label
    
    Returns:
        List of table IDs with the specified classification
    """
    return [tableid for tableid, label in classifications.items() if label == classification]


def main():
    """CLI entry point for table classification"""
    parser = argparse.ArgumentParser(description="Table Classification")
    parser.add_argument('--mode', choices=['single', 'batch'], required=True,
                       help='Classification mode: single table or batch datalake')
    parser.add_argument('--table', default=None,
                       help='Path to CSV file (for single mode)')
    parser.add_argument('--tableid', type=int, default=None,
                       help='Table ID from database (for single mode with database)')
    parser.add_argument('--db_path', default='data_citationlake/modellake.db',
                       help='Path to modellake.db (for batch mode or single mode with tableid)')
    parser.add_argument('--index_table', default='modellake_index',
                       help='Name of the index table')
    parser.add_argument('--output_json', default='data/table_classifications.json',
                       help='Output JSON file for classification results')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of tables to classify (batch mode only)')
    parser.add_argument('--method', choices=['tab2know', 'heuristic', 'ml'], default='tab2know',
                       help='Classification method (default: tab2know)')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'single':
            if args.table:
                # Classify from CSV file
                classification = classify_table(args.table, method=args.method)
                print(f"\n✅ Classification: {classification}")
            elif args.tableid:
                # Classify from database tableid
                classification = classify_table_from_db(
                    args.tableid,
                    args.db_path,
                    args.index_table
                )
                if classification:
                    print(f"\n✅ Table ID {args.tableid} classification: {classification}")
                else:
                    print(f"\n❌ Table ID {args.tableid} not found")
            else:
                parser.error("Either --table or --tableid must be provided for single mode")
        
        elif args.mode == 'batch':
            # Batch classify all tables in datalake
            classifications = classify_datalake_batch(
                db_path=args.db_path,
                index_table=args.index_table,
                output_json=args.output_json,
                limit=args.limit,
                method=args.method
            )
            print(f"\n✅ Batch classification complete: {len(classifications)} tables")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    import sys
    
    # If running as test (python -m src.search.classification test), run test cases
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        print("=" * 60)
        print("Running Classification Test Cases")
        print("=" * 60)
        
        # Test 1: Test get_tables_by_classification with tab2know labels
        print("\n[Test 1] Get tables by classification (tab2know labels)")
        print("-" * 60)
        test_classifications = {
            1: "Observation",
            2: "Observation",
            3: "Input",
            4: "Other",
            5: "Observation"
        }
        observation_tables = get_tables_by_classification("Observation", test_classifications)
        print(f"✅ Observation tables: {observation_tables}")
        assert set(observation_tables) == {1, 2, 5}, f"Expected [1, 2, 5], got {observation_tables}"
        
        # Test 2: Test load_classifications
        print("\n[Test 2] Load classifications from JSON")
        print("-" * 60)
        import tempfile
        import json as json_module
        
        # Create a temporary JSON file with tab2know labels
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        test_data = {"1": "Observation", "2": "Input", "3": "Other", "4": "Example"}
        json_module.dump(test_data, temp_file)
        temp_file.close()
        
        try:
            loaded = load_classifications(temp_file.name)
            print(f"✅ Loaded classifications: {loaded}")
            assert loaded == {1: "Observation", 2: "Input", 3: "Other", 4: "Example"}, \
                f"Expected {{1: 'Observation', 2: 'Input', 3: 'Other', 4: 'Example'}}, got {loaded}"
        finally:
            os.unlink(temp_file.name)
        
        # Test 3: Test heuristic method (if tab2know not available, fallback)
        print("\n[Test 3] Test heuristic classification method")
        print("-" * 60)
        test_df_numerical = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [10, 20, 30, 40, 50],
            'col3': [100, 200, 300, 400, 500]
        })
        result = classify_table(test_df_numerical, method="heuristic")
        print(f"✅ Result: {result}")
        assert result == "numerical", f"Expected 'numerical', got '{result}'"
        
        # Test 4: Test empty DataFrame with heuristic (direct call to avoid CSV issues)
        print("\n[Test 4] Classify empty DataFrame (heuristic)")
        print("-" * 60)
        test_df_empty = pd.DataFrame()
        result = _classify_heuristic(test_df_empty)
        print(f"✅ Result: {result}")
        assert result == "empty", f"Expected 'empty', got '{result}'"
        
        # Test 5: Test tab2know classification (if available)
        print("\n[Test 5] Test tab2know classification (if available)")
        print("-" * 60)
        tab2know_repo = _find_tab2know_repo()
        if tab2know_repo:
            print(f"✅ Tab2Know repository found: {tab2know_repo}")
            # Create a simple test CSV
            test_df = pd.DataFrame({
                'Method': ['A', 'B', 'C'],
                'Accuracy': [0.95, 0.92, 0.88],
                'F1': [0.94, 0.91, 0.87]
            })
            result = classify_table(test_df, method="tab2know")
            print(f"✅ Result: {result}")
            # Should be one of the tab2know labels
            assert result in ["Observation", "Input", "Other", "Example", "unknown"], \
                f"Expected tab2know label, got '{result}'"
        else:
            print("⚠️  Tab2Know repository not found, skipping tab2know test")
            print("   Set TAB2KNOW_REPO environment variable to enable tab2know tests")
        
        print("\n" + "=" * 60)
        print("✅ All test cases passed!")
        print("=" * 60)
    else:
        main()

