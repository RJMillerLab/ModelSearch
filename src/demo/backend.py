"""
Backend API for ModelSearch Demo

Runs search commands, saves results, and provides REST API with real-time progress logs.
"""

import os
import sys
import json
import uuid
import threading
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.search import (
    search_query2modelcard,
    search_card2card,
    search_card2tab2card,
    get_tables_for_model,
    load_relationship_parquet
)
from src.integration.table_integration import integrate_tables_from_search_results


app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Default paths
DEFAULT_EMB_NPZ = "data/card2card_embeddings.npz"
DEFAULT_FAISS_INDEX = "data/card2card.faiss"
DEFAULT_SCHEMA_LOG = "data_citationlake/logs/parquet_schema.log"
DEFAULT_RELATIONSHIP_PARQUET = "data_citationlake/processed/modelcard_step3_dedup.parquet"
DEFAULT_DB_PATH = "data_citationlake/modellake.db"

# Job storage (in production, use Redis or database)
jobs = {}


class JobLogger:
    """Thread-safe logger for job progress"""
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.logs = []
        self.lock = threading.Lock()
        self.status = "pending"  # pending, running, completed, error
        self.results = None
    
    def log(self, message: str):
        """Add log message with timestamp"""
        with self.lock:
            now = datetime.now()
            timestamp = now.isoformat()
            # Format timestamp for display: YYYY-MM-DD HH:MM:SS.mmm
            timestamp_display = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            self.logs.append({"timestamp": timestamp, "message": message})
            print(f"[{timestamp_display}] [{self.job_id}] {message}")
    
    def get_logs(self) -> List[Dict]:
        """Get all logs"""
        with self.lock:
            return self.logs.copy()
    
    def set_status(self, status: str):
        """Set job status"""
        with self.lock:
            self.status = status
    
    def set_results(self, results: Dict):
        """Set job results"""
        with self.lock:
            self.results = results
            self.status = "completed"


def run_search_pipeline(job_id: str, query: Optional[str] = None, top_k: int = 20, model_id: Optional[str] = None, table_search_k: Optional[int] = None):
    """Run the complete search pipeline in background
    
    Args:
        job_id: Job identifier
        query: Text query (for query mode)
        top_k: Number of results to return
        model_id: Model ID (for modelid mode)
        table_search_k: Number of tables to retrieve in Card2Tab2Card search (defaults to top_k * 2)
    """
    logger = jobs[job_id]
    
    # Record start time
    start_time = time.time()
    
    try:
        logger.set_status("running")
        logger.log("Starting search pipeline...")
        
        # Step 1: Get model ID (either from query or directly provided)
        if model_id:
            # Direct model ID mode
            logger.log(f"Mode: ModelID → Search (direct)")
            logger.log(f"Model ID: {model_id}")
            
            # Verify model ID exists in the index
            logger.log("Step 1: Verifying model ID exists in index...")
            try:
                import numpy as np
                data = np.load(DEFAULT_EMB_NPZ)
                ids = data['ids'].tolist()
                if model_id not in ids:
                    raise ValueError(f"Model ID '{model_id}' not found in corpus. Please check the model ID.")
                logger.log(f"✅ Model ID verified: {model_id}")
            except Exception as e:
                logger.log(f"❌ Error verifying model ID: {str(e)}")
                raise
        else:
            # Query mode: Extract model card from query
            logger.log(f"Mode: Query → ModelCard → Search")
            logger.log(f"Query: {query}")
            logger.log("Step 1: Extracting model card from query...")
            model_results = search_query2modelcard(
                query=query,
                emb_npz=DEFAULT_EMB_NPZ,
                faiss_index=DEFAULT_FAISS_INDEX,
                top_k=1,  # Get top 1 as the query model
                device="cuda",
                output_json=None
            )
            
            if not model_results:
                raise ValueError("No model card found for query")
            
            model_id = model_results[0]
            logger.log(f"✅ Extracted model: {model_id}")
        
        # Step 2: Check if model has tables for Card2Tab2Card
        logger.log("Step 2: Checking if model has tables...")
        model_tables = []
        has_tables = False
        relationship_df = None
        
        try:
            # Try to use CitationLake first, fallback to relationship_parquet if not available
            try:
                from src.search.card2tab2card import USE_CITATIONLAKE_GET_FROM
            except ImportError:
                USE_CITATIONLAKE_GET_FROM = False
            
            if USE_CITATIONLAKE_GET_FROM:
                # Try CitationLake approach
                model_tables = get_tables_for_model(
                    model_id=model_id,
                    schema_log_path=DEFAULT_SCHEMA_LOG,
                    use_citationlake=True
                )
            else:
                # CitationLake not available, use relationship_parquet
                logger.log("⚠️  CitationLake not available, using relationship_parquet...")
                relationship_df = load_relationship_parquet(DEFAULT_RELATIONSHIP_PARQUET)
                model_tables = get_tables_for_model(
                    model_id=model_id,
                    relationship_df=relationship_df,
                    schema_log_path=DEFAULT_SCHEMA_LOG,
                    use_citationlake=False
                )
            
            if not model_tables:
                logger.log(f"⚠️  Model {model_id} has no tables - Card2Tab2Card pipeline will be skipped")
                logger.log(f"   This model card does not have associated tables, so table-based search cannot be performed.")
                has_tables = False
            else:
                logger.log(f"✅ Model has {len(model_tables)} tables - Card2Tab2Card pipeline can proceed")
                has_tables = True
        except Exception as e:
            logger.log(f"⚠️  Error checking tables for model: {str(e)}")
            logger.log(f"   Card2Tab2Card pipeline may fail")
            has_tables = False
            model_tables = []
        
        # Prepare query CSV for Card2Tab2Card
        query_csv = None
        
        # Try to find a CSV file from the model's tables
        if has_tables and model_tables:
            # Try to use one of the model's actual tables
            for table_path in model_tables[:5]:  # Try first 5 tables
                # Check if it's a full path or basename
                if os.path.exists(table_path):
                    query_csv = table_path
                    break
                else:
                    # Try to find in common locations
                    for base_dir in [
                        "data_citationlake/processed/deduped_hugging_csvs",
                        "data_citationlake/processed/deduped_github_csvs",
                        "data_citationlake/processed/tables_output"
                    ]:
                        full_path = os.path.join(base_dir, os.path.basename(table_path))
                        if os.path.exists(full_path):
                            query_csv = full_path
                            break
                    if query_csv:
                        break
        
        # Fallback to default CSV if model tables not found
        if not query_csv:
            default_csvs = [
                "data_citationlake/processed/deduped_github_csvs/0021c79d4e1a37579ca87328864d67a5_table_0.csv",
                "data_citationlake/processed/deduped_hugging_csvs/0000e35dae_table1.csv"
            ]
            for csv_path in default_csvs:
                if os.path.exists(csv_path):
                    query_csv = csv_path
                    break
        
        if not query_csv:
            logger.log("⚠️  No CSV found, using model's table basenames as query")
            query_csv = None
        
        # Load CSV once for all search types
        import pandas as pd
        if query_csv:
            query_df = pd.read_csv(query_csv)
            logger.log(f"✅ Loaded CSV: {query_csv} ({len(query_df)} rows, {len(query_df.columns)} columns)")
        else:
            query_df = None
        
        # Step 3: Run Card2Card and Card2Tab2Card in parallel
        logger.log("Step 3: Running Card2Card and Card2Tab2Card pipelines in parallel...")
        
        def run_card2card():
            """Run Card2Card pipeline"""
            logger.log("  [Card2Card] Starting dense semantic search...")
            try:
                # Increase topk slightly to get more results (e.g., topk=3 -> 5-10 results)
                # Use max(top_k * 2, top_k + 5) to ensure we get more results
                expanded_topk = max(top_k * 2, top_k + 5)
                logger.log(f"  ℹ️  [Card2Card] Using expanded top_k: {expanded_topk} (requested: {top_k})")
                results = search_card2card(
                    model_id=model_id,
                    emb_npz=DEFAULT_EMB_NPZ,
                    faiss_index=DEFAULT_FAISS_INDEX,
                    top_k=expanded_topk,
                    output_json=None
                )
                # Limit to requested top_k for final results
                final_results = results[:top_k]
                logger.log(f"  ✅ [Card2Card] Found {len(results)} results (returning top {len(final_results)})")
                return final_results
            except Exception as e:
                logger.log(f"  ❌ [Card2Card] Error: {str(e)}")
                return {"error": str(e)}
        
        def run_card2tab2card_search(search_type_name, query_parsed, table_search_k=None):
            """Run one Card2Tab2Card search type"""
            logger.log(f"  [Card2Tab2Card-{search_type_name}] Starting...")
            try:
                # Use a temporary JSON file to capture intermediate results
                import tempfile
                import json
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                    tmp_json_path = tmp_file.name
                
                # Log query details for debugging
                if search_type_name == 'single_column':
                    logger.log(f"  ℹ️  [Card2Tab2Card-{search_type_name}] Query: {len(query_parsed) if query_parsed else 0} values")
                elif search_type_name == 'keyword':
                    logger.log(f"  ℹ️  [Card2Tab2Card-{search_type_name}] Query: {len(query_parsed) if query_parsed else 0} keywords")
                elif search_type_name == 'unionable':
                    logger.log(f"  ℹ️  [Card2Tab2Card-{search_type_name}] Query: DataFrame with {len(query_parsed) if query_parsed is not None else 0} rows")
                
                # Use provided table_search_k or default to top_k * 2 for table search
                # This allows more tables to be retrieved, then we limit modelcards to top_k
                if table_search_k is None:
                    table_search_k = top_k * 2  # Get more tables, then filter to top_k modelcards
                
                logger.log(f"  ℹ️  [Card2Tab2Card-{search_type_name}] Table search k: {table_search_k}, ModelCard k: {top_k}")
                
                results = search_card2tab2card(
                    model_id=model_id,
                    relationship_parquet=DEFAULT_RELATIONSHIP_PARQUET,
                    query=query_parsed,
                    search_type=search_type_name,
                    k=top_k,  # Legacy parameter for backward compatibility
                    table_search_k=table_search_k,  # Control table search topk
                    modelcard_k=top_k,  # Control final modelcard topk
                    schema_log_path=DEFAULT_SCHEMA_LOG,
                    use_citationlake=True,
                    output_json=tmp_json_path,
                    db_path=DEFAULT_DB_PATH
                )
                
                # Read intermediate results from JSON
                intermediate_data = {}
                try:
                    # Wait a bit for file to be written (in case of async issues)
                    import time
                    time.sleep(0.1)
                    
                    # Check if file exists and is not empty
                    if os.path.exists(tmp_json_path):
                        file_size = os.path.getsize(tmp_json_path)
                        if file_size > 0:
                            with open(tmp_json_path, 'r', encoding='utf-8') as f:
                                content = f.read().strip()
                                if content:  # Check if file has content
                                    full_results = json.loads(content)
                                    intermediate_data = full_results.get('intermediate', {})
                                    logger.log(f"  ℹ️  [Card2Tab2Card-{search_type_name}] Loaded intermediate data from {tmp_json_path} ({file_size} bytes)")
                                else:
                                    logger.log(f"  ⚠️  [Card2Tab2Card-{search_type_name}] JSON file is empty")
                        else:
                            logger.log(f"  ⚠️  [Card2Tab2Card-{search_type_name}] JSON file exists but is empty (0 bytes)")
                    else:
                        logger.log(f"  ⚠️  [Card2Tab2Card-{search_type_name}] JSON file does not exist: {tmp_json_path}")
                    
                    # Clean up temp file
                    if os.path.exists(tmp_json_path):
                        os.unlink(tmp_json_path)
                except json.JSONDecodeError as e:
                    logger.log(f"  ⚠️  [Card2Tab2Card-{search_type_name}] JSON decode error: {str(e)}")
                    # Try to read file content for debugging
                    if os.path.exists(tmp_json_path):
                        try:
                            with open(tmp_json_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                logger.log(f"  ⚠️  File content (first 200 chars): {content[:200]}")
                        except:
                            pass
                        os.unlink(tmp_json_path)
                except Exception as e:
                    logger.log(f"  ⚠️  [Card2Tab2Card-{search_type_name}] Could not read intermediate data: {str(e)}")
                    if os.path.exists(tmp_json_path):
                        os.unlink(tmp_json_path)
                
                logger.log(f"  ✅ [Card2Tab2Card-{search_type_name}] Found {len(results)} results")
                return search_type_name, {
                    "model_ids": results,
                    "intermediate": intermediate_data
                }
            except Exception as e:
                logger.log(f"  ❌ [Card2Tab2Card-{search_type_name}] Error: {str(e)}")
                return search_type_name, {"error": str(e)}
        
        # Prepare queries for all search types
        # Follow Blend_internal logic from README examples:
        # - keyword: use headers (column names) - rowid=-1 in index
        # - single_column: use values from first column (Seekers.SC(dataset[clm_name], k) uses single column)
        # - unionable: use entire DataFrame
        # - complex: use entire DataFrame as examples, auto-detect target column
        queries_parsed = {}
        if query_df is not None:
            # For single_column: use values from first column
            # From README: Seekers.SC(dataset[clm_name], k) - uses single column
            # From ComplexSearch: Seekers.SC(examples[examples.columns[0]], k) - uses first column
            first_col = query_df.columns[0]
            queries_parsed['single_column'] = query_df[first_col].dropna().astype(str).tolist()
            logger.log(f"ℹ️  Single_column search using first column '{first_col}' with {len(queries_parsed['single_column'])} values")
            
            # For keyword: use headers (column names) - consistent with Blend_internal
            # Blend_internal uses rowid=-1 which represents headers in the index
            headers = [str(col).lower().strip() for col in query_df.columns]
            # Filter out empty headers (consistent with Blend_internal self_keyword.py)
            headers = [h for h in headers if h]
            queries_parsed['keyword'] = headers
            logger.log(f"ℹ️  Keyword search using {len(headers)} headers: {headers[:5]}{'...' if len(headers) > 5 else ''}")
            
            # For unionable: use entire DataFrame (same as Blend_internal)
            queries_parsed['unionable'] = query_df
            
            # For complex: use entire DataFrame as examples
            # ComplexSearch will auto-detect numeric column as target
            queries_parsed['complex'] = query_df
            logger.log(f"ℹ️  Complex search using DataFrame with {len(query_df)} rows and {len(query_df.columns)} columns")
            # Try to identify potential target column
            numeric_cols = query_df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                logger.log(f"   Potential target columns (numeric): {numeric_cols[:3]}{'...' if len(numeric_cols) > 3 else ''}")
            
            # For correlation: use entire DataFrame, will extract source and target columns
            queries_parsed['correlation'] = query_df
            logger.log(f"ℹ️  Correlation search using DataFrame with {len(query_df)} rows and {len(query_df.columns)} columns")
            if numeric_cols:
                logger.log(f"   Will use first column as source, first numeric column '{numeric_cols[0]}' as target")
            else:
                logger.log(f"   ⚠️  No numeric column found - correlation search may not work well")
        else:
            queries_parsed['single_column'] = None
            queries_parsed['keyword'] = None
            queries_parsed['unionable'] = None
            queries_parsed['complex'] = None
            queries_parsed['correlation'] = None
        
        # Run all searches in parallel using ThreadPoolExecutor
        card2card_results = None
        card2tab2card_all = {}
        
        # Only run Card2Tab2Card if model has tables
        if not has_tables:
            logger.log("⚠️  Skipping Card2Tab2Card pipeline - model has no tables")
            # Only run Card2Card
            card2card_results = run_card2card()
            # Set empty results for Card2Tab2Card
            card2tab2card_all = {
                'single_column': [],
                'keyword': [],
                'unionable': [],
                'complex': [],
                'correlation': []
            }
        else:
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Submit all tasks
                futures = {}
                
                # Submit Card2Card
                futures['card2card'] = executor.submit(run_card2card)
                
                # Submit all Card2Tab2Card search types (run all: single_column, keyword, unionable, complex, correlation)
                all_search_types = ['single_column', 'keyword', 'unionable', 'complex', 'correlation']
                logger.log(f"  ℹ️  Running {len(all_search_types)} search types: {', '.join(all_search_types)}")
                for search_type_name in all_search_types:
                    query_parsed = queries_parsed[search_type_name]
                    # Skip if query is None (e.g., no CSV loaded)
                    if query_parsed is None:
                        logger.log(f"  ⚠️  Skipping {search_type_name} - no query data available")
                        continue
                    futures[search_type_name] = executor.submit(
                        run_card2tab2card_search, 
                        search_type_name, 
                        query_parsed,
                        table_search_k  # Pass table_search_k parameter
                    )
            
            # Collect results as they complete
            for future in as_completed(futures.values()):
                try:
                    result = future.result()
                    if isinstance(result, tuple):
                        # Card2Tab2Card result
                        search_type_name, results = result
                        card2tab2card_all[search_type_name] = results
                    else:
                        # Card2Card result
                        card2card_results = result
                except Exception as e:
                    logger.log(f"  ❌ Future error: {str(e)}")
        
        # Ensure we got card2card results
        if card2card_results is None or isinstance(card2card_results, dict) and "error" in card2card_results:
            raise ValueError("Card2Card search failed")
        
        # Step 4: Compare results
        logger.log("Step 4: Comparing results...")
        card2card_set = set(card2card_results)
        comparison = {}
        
        for search_type_name, tab2card_results in card2tab2card_all.items():
            # Handle both old format (list) and new format (dict with model_ids and intermediate)
            if isinstance(tab2card_results, dict) and "model_ids" in tab2card_results:
                model_ids_list = tab2card_results["model_ids"]
            elif isinstance(tab2card_results, list):
                model_ids_list = tab2card_results
            else:
                model_ids_list = []
            
            if model_ids_list:
                tab2card_set = set(model_ids_list)
                overlap = card2card_set & tab2card_set
                comparison[search_type_name] = {
                    "card2card_count": len(card2card_set),
                    "card2tab2card_count": len(tab2card_set),
                    "overlap_count": len(overlap),
                    "overlap_ratio": len(overlap) / len(card2card_set) if card2card_set else 0,
                    "overlap_models": list(overlap),
                    "card2card_only": list(card2card_set - tab2card_set),
                    "card2tab2card_only": list(tab2card_set - card2card_set)
                }
        
        # Add hyperlinks to model IDs
        def add_hyperlinks(model_list):
            """Add HuggingFace hyperlinks to model IDs"""
            if isinstance(model_list, list):
                return [
                    {
                        "model_id": model_id,
                        "url": f"https://huggingface.co/{model_id}"
                    }
                    for model_id in model_list
                ]
            return model_list
        
        card2card_results_with_links = add_hyperlinks(card2card_results)
        card2tab2card_all_with_links = {}
        for search_type, results in card2tab2card_all.items():
            # Handle new format with intermediate data
            if isinstance(results, dict) and "model_ids" in results:
                card2tab2card_all_with_links[search_type] = {
                    "model_ids": add_hyperlinks(results["model_ids"]),
                    "intermediate": results.get("intermediate", {})
                }
            elif isinstance(results, list):
                card2tab2card_all_with_links[search_type] = add_hyperlinks(results)
            else:
                card2tab2card_all_with_links[search_type] = results
        
        # Save results with timestamp-based filename
        # Use query time (start_time) for timestamp to ensure consistency
        timestamp_str = datetime.fromtimestamp(start_time).strftime('%Y%m%d_%H%M%S')
        
        # Create descriptive filename based on mode
        if query:
            # Query mode: use first 30 chars of query (sanitized)
            query_safe = "".join(c for c in query[:30] if c.isalnum() or c in (' ', '-', '_')).strip()
            query_safe = query_safe.replace(' ', '_')
            filename = f'search_{timestamp_str}_{query_safe}.json'
        else:
            # ModelID mode: use model_id (sanitized)
            model_id_safe = model_id.replace('/', '_').replace('\\', '_')
            filename = f'search_{timestamp_str}_{model_id_safe}.json'
        
        # Limit filename length
        if len(filename) > 150:
            filename = filename[:150] + '.json'
        
        output_json = os.path.join('data', filename)
        os.makedirs('data', exist_ok=True)
        
        results_data = {
            "job_id": job_id,
            "query": query,
            "model_id": model_id,
            "model_url": f"https://huggingface.co/{model_id}",
            "top_k": top_k,
            "table_search_k": table_search_k,
            "card2card_results": card2card_results_with_links,
            "card2tab2card_results": card2tab2card_all_with_links,
            "comparison": comparison,
            "timestamp": datetime.fromtimestamp(start_time).isoformat(),
            "timestamp_str": timestamp_str,
            "filename": filename
        }
        
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        
        logger.log(f"✅ Results saved to {output_json}")
        
        # Also save with job_id for backward compatibility
        job_id_json = os.path.join('data', f'compare_search_{job_id}.json')
        with open(job_id_json, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        
        # Calculate and log running time
        end_time = time.time()
        running_time = end_time - start_time
        minutes = int(running_time // 60)
        seconds = int(running_time % 60)
        milliseconds = int((running_time % 1) * 1000)
        
        if minutes > 0:
            logger.log(f"⏱️  Total running time: {minutes}m {seconds}s {milliseconds}ms")
        else:
            logger.log(f"⏱️  Total running time: {seconds}s {milliseconds}ms")
        
        # Add running time to results
        results_data["running_time_seconds"] = round(running_time, 3)
        results_data["running_time_formatted"] = f"{minutes}m {seconds}s {milliseconds}ms" if minutes > 0 else f"{seconds}s {milliseconds}ms"
        
        logger.set_results(results_data)
        logger.log("Pipeline completed successfully!")
        
    except Exception as e:
        # Calculate running time even on error
        end_time = time.time()
        running_time = end_time - start_time
        minutes = int(running_time // 60)
        seconds = int(running_time % 60)
        milliseconds = int((running_time % 1) * 1000)
        
        if minutes > 0:
            logger.log(f"⏱️  Running time before error: {minutes}m {seconds}s {milliseconds}ms")
        else:
            logger.log(f"⏱️  Running time before error: {seconds}s {milliseconds}ms")
        
        logger.log(f"❌ Error: {str(e)}")
        logger.set_status("error")
        import traceback
        logger.log(traceback.format_exc())


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok"})


@app.route('/api/table-preview', methods=['GET'])
def get_table_preview():
    """Get preview of CSV table (first 5 rows, first 5 columns) from modellake.db or file path"""
    table_path = request.args.get('path')
    if not table_path:
        return jsonify({"status": "error", "message": "path parameter is required"}), 400
    
    # Extract basename from path
    basename = os.path.basename(table_path)
    
    # Log the request
    print(f"\n{'='*60}")
    print(f"🔍 Table Preview Request")
    print(f"{'='*60}")
    print(f"Input path: {table_path}")
    print(f"Basename: {basename}")
    
    # Build comprehensive list of possible base directories
    # Include both data_citationlake and CitationLake paths
    possible_base_dirs = [
        # data_citationlake paths (current structure)
        "data_citationlake/processed/deduped_hugging_csvs",
        "data_citationlake/processed/deduped_github_csvs",
        "data_citationlake/processed/tables_output",
        # CitationLake paths (if CitationLake is in parent directory)
        "../CitationLake/data/processed/deduped_hugging_csvs",
        "../CitationLake/data/processed/deduped_github_csvs",
        "../CitationLake/data/processed/tables_output",
        # Alternative CitationLake paths
        "../../CitationLake/data/processed/deduped_hugging_csvs",
        "../../CitationLake/data/processed/deduped_github_csvs",
        "../../CitationLake/data/processed/tables_output",
    ]
    
    # Try to find CSV path from modellake.db first to get hints
    csv_path = None
    table_type_hint = None
    db_info = None
    
    # Method 1: Try to get path hint from DB
    print(f"\n📊 Method 1: Querying modellake.db...")
    try:
        import duckdb
        if os.path.exists(DEFAULT_DB_PATH):
            print(f"   DB path: {DEFAULT_DB_PATH}")
            con = duckdb.connect(DEFAULT_DB_PATH, read_only=True)
            try:
                # Query modellake.db to get table_group and table_type for this filename
                query = """
                    SELECT DISTINCT filename, table_group, table_type 
                    FROM modellake_index 
                    WHERE filename = ? AND rowid = -1
                    LIMIT 1
                """
                result = con.execute(query, [basename]).fetchone()
                
                if result:
                    filename, table_group, table_type = result
                    table_type_hint = table_type
                    db_info = {
                        "filename": filename,
                        "table_group": table_group,
                        "table_type": table_type
                    }
                    print(f"   ✅ Found in DB: filename={filename}, table_group={table_group}, table_type={table_type}")
                    # table_type can be 'ori', 'str', 'tr', etc.
                    # 'ori' usually means original files in deduped directories
                else:
                    print(f"   ⚠️  Not found in DB for basename: {basename}")
            finally:
                con.close()
        else:
            print(f"   ⚠️  DB file not found: {DEFAULT_DB_PATH}")
    except Exception as e:
        # If DB query fails, fall back to file system search
        print(f"   ⚠️  DB query error (non-fatal): {str(e)}")
        pass
    
    # Method 2: Try to find file using path-based strategies
    print(f"\n📁 Method 2: Searching file system...")
    
    # Strategy 1: If table_path is already a valid absolute or relative path, try it first
    if os.path.exists(table_path):
        csv_path = table_path
        print(f"   ✅ Found at provided path: {table_path}")
    else:
        print(f"   ⚠️  Provided path does not exist, searching alternatives...")
        
        # Strategy 2: Try to infer directory from table_path if it contains path hints
        # Check if table_path contains hints like "hugging", "github", etc.
        path_lower = table_path.lower()
        if "hugging" in path_lower:
            print(f"   🔍 Path contains 'hugging', searching hugging directories...")
            # Try hugging directories first
            for base_dir in [d for d in possible_base_dirs if "hugging" in d.lower()]:
                if base_dir.startswith('../'):
                    abs_base_dir = os.path.abspath(os.path.join(os.getcwd(), base_dir))
                else:
                    abs_base_dir = os.path.abspath(base_dir)
                full_path = os.path.join(abs_base_dir, basename)
                print(f"      Trying: {full_path}")
                if os.path.exists(full_path):
                    csv_path = full_path
                    print(f"      ✅ Found: {full_path}")
                    break
        elif "github" in path_lower:
            print(f"   🔍 Path contains 'github', searching github directories...")
            # Try github directories first
            for base_dir in [d for d in possible_base_dirs if "github" in d.lower()]:
                if base_dir.startswith('../'):
                    abs_base_dir = os.path.abspath(os.path.join(os.getcwd(), base_dir))
                else:
                    abs_base_dir = os.path.abspath(base_dir)
                full_path = os.path.join(abs_base_dir, basename)
                print(f"      Trying: {full_path}")
                if os.path.exists(full_path):
                    csv_path = full_path
                    print(f"      ✅ Found: {full_path}")
                    break
        
        # Strategy 3: If table_type hint is available from DB, prioritize accordingly
        if not csv_path and table_type_hint:
            print(f"   🔍 Using DB hint (table_type={table_type_hint})...")
            if table_type_hint == 'ori':
                # Original files are usually in deduped directories
                for base_dir in [d for d in possible_base_dirs if "deduped" in d.lower()]:
                    if base_dir.startswith('../'):
                        abs_base_dir = os.path.abspath(os.path.join(os.getcwd(), base_dir))
                    else:
                        abs_base_dir = os.path.abspath(base_dir)
                    full_path = os.path.join(abs_base_dir, basename)
                    print(f"      Trying: {full_path}")
                    if os.path.exists(full_path):
                        csv_path = full_path
                        print(f"      ✅ Found: {full_path}")
                        break
            elif table_type_hint in ['str', 'tr']:
                # Transformed files might be in tables_output
                for base_dir in [d for d in possible_base_dirs if "tables_output" in d.lower()]:
                    if base_dir.startswith('../'):
                        abs_base_dir = os.path.abspath(os.path.join(os.getcwd(), base_dir))
                    else:
                        abs_base_dir = os.path.abspath(base_dir)
                    full_path = os.path.join(abs_base_dir, basename)
                    print(f"      Trying: {full_path}")
                    if os.path.exists(full_path):
                        csv_path = full_path
                        print(f"      ✅ Found: {full_path}")
                        break
        
        # Strategy 4: Exhaustive search through all possible directories
        if not csv_path:
            print(f"   🔍 Exhaustive search through all directories...")
            for base_dir in possible_base_dirs:
                # Convert relative paths to absolute
                if base_dir.startswith('../'):
                    # Try to resolve relative to current working directory
                    abs_base_dir = os.path.abspath(os.path.join(os.getcwd(), base_dir))
                else:
                    abs_base_dir = os.path.abspath(base_dir)
                
                full_path = os.path.join(abs_base_dir, basename)
                if os.path.exists(full_path):
                    csv_path = full_path
                    print(f"      ✅ Found: {full_path}")
                    break
                # Don't print every failed attempt to avoid too much output
    
    if not csv_path or not os.path.exists(csv_path):
        # Provide detailed error message with searched paths
        print(f"\n❌ File not found after all search attempts")
        searched_paths = [os.path.abspath(d) if not d.startswith('../') else os.path.abspath(os.path.join(os.getcwd(), d)) for d in possible_base_dirs[:6]]
        print(f"   Searched in {len(searched_paths)} directories")
        return jsonify({
            "status": "error",
            "message": f"CSV file not found: {basename}",
            "searched_directories": searched_paths[:3],  # Show first 3 for brevity
            "db_info": db_info,
            "hint": "Make sure the file exists in one of the processed directories"
        }), 404
    
    print(f"\n✅ File found: {csv_path}")
    print(f"{'='*60}\n")
    
    try:
        import pandas as pd
        print(f"📖 Reading CSV file...")
        # Read first 5 rows and first 5 columns
        df = pd.read_csv(csv_path, nrows=5)
        print(f"   ✅ Loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Limit to first 5 columns
        df_preview = df.iloc[:, :5]
        print(f"   ✅ Preview: {len(df_preview)} rows, {len(df_preview.columns)} columns")
        
        # Convert to simple HTML table
        html_table = "<table style='width: 100%; border-collapse: collapse; font-size: 11px;'>"
        # Header
        headers = df_preview.columns.tolist()
        html_table += "<tr>"
        for h in headers:
            # Escape HTML special characters
            h_escaped = str(h).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            html_table += f"<th style='border: 1px solid #dee2e6; padding: 4px; background: #f8f9fa; text-align: left;'>{h_escaped}</th>"
        html_table += "</tr>"
        # Rows
        for _, row in df_preview.iterrows():
            html_table += "<tr>"
            for val in row.values:
                # Escape HTML special characters and handle NaN
                val_str = str(val) if pd.notna(val) else ''
                val_escaped = val_str.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                html_table += f"<td style='border: 1px solid #dee2e6; padding: 4px;'>{val_escaped}</td>"
            html_table += "</tr>"
        html_table += "</table>"
        
        print(f"✅ Successfully generated preview HTML")
        
        return jsonify({
            "status": "success",
            "table_path": csv_path,
            "rows": len(df_preview),
            "columns": len(df_preview.columns),
            "html": html_table,
            "db_info": db_info  # Include DB info if available
        })
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ Error reading CSV: {str(e)}")
        print(f"   Traceback: {error_trace}")
        return jsonify({
            "status": "error",
            "message": f"Error reading CSV: {str(e)}",
            "table_path": csv_path,
            "db_info": db_info
        }), 500


@app.route('/api/search', methods=['POST'])
def search():
    """Main search endpoint - starts pipeline or loads from saved results (mimic mode)"""
    try:
        data = request.json or {}
        search_mode = data.get('search_mode', 'new')  # 'new' or 'mimic'
        
        # Mimic mode: load from saved results
        if search_mode == 'mimic':
            filename = data.get('filename')
            if not filename:
                return jsonify({"status": "error", "message": "filename is required for mimic mode"}), 400
            
            # Try to find the file
            file_path = os.path.join('data', filename)
            if not os.path.exists(file_path):
                return jsonify({
                    "status": "error",
                    "message": f"Saved results file not found: {filename}"
                }), 404
            
            # Load the saved results
            with open(file_path, 'r', encoding='utf-8') as f:
                saved_results = json.load(f)
            
            # Create a new job_id for this mimic session
            job_id = str(uuid.uuid4())
            jobs[job_id] = JobLogger(job_id)
            jobs[job_id].set_results(saved_results)
            jobs[job_id].set_status("completed")
            jobs[job_id].log(f"✅ Loaded saved results from {filename}")
            jobs[job_id].log(f"   Original query: {saved_results.get('query', 'N/A')}")
            jobs[job_id].log(f"   Original model_id: {saved_results.get('model_id', 'N/A')}")
            jobs[job_id].log(f"   Original timestamp: {saved_results.get('timestamp', 'N/A')}")
            
            return jsonify({
                "status": "completed",
                "job_id": job_id,
                "message": "Loaded saved results (mimic mode)",
                "results": saved_results
            })
        
        # New search mode: run actual search
        mode = data.get('mode', 'query')  # 'query' or 'modelid'
        top_k = data.get('top_k', 20)
        table_search_k = data.get('table_search_k', None)  # Optional: defaults to top_k * 2 in pipeline
        
        if mode == 'query':
            query = data.get('query')
            if not query:
                return jsonify({"status": "error", "message": "query is required for query mode"}), 400
            model_id = None
        elif mode == 'modelid':
            model_id = data.get('model_id')
            if not model_id:
                return jsonify({"status": "error", "message": "model_id is required for modelid mode"}), 400
            query = None
        else:
            return jsonify({"status": "error", "message": f"Invalid mode: {mode}. Must be 'query' or 'modelid'"}), 400
        
        # Create job
        job_id = str(uuid.uuid4())
        jobs[job_id] = JobLogger(job_id)
        
        # Start pipeline in background thread
        thread = threading.Thread(
            target=run_search_pipeline,
            args=(job_id, query, top_k, model_id, table_search_k)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "status": "started",
            "job_id": job_id,
            "message": "Search pipeline started"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/saved-searches', methods=['GET'])
def list_saved_searches():
    """List all saved search results"""
    try:
        data_dir = 'data'
        if not os.path.exists(data_dir):
            return jsonify({
                "status": "success",
                "searches": []
            })
        
        # Find all search result files
        search_files = []
        for filename in os.listdir(data_dir):
            if filename.startswith('search_') and filename.endswith('.json'):
                file_path = os.path.join(data_dir, filename)
                try:
                    # Read metadata without loading full file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Get file stats
                    stat = os.stat(file_path)
                    
                    search_files.append({
                        "filename": filename,
                        "query": data.get('query', ''),
                        "model_id": data.get('model_id', ''),
                        "timestamp": data.get('timestamp', ''),
                        "timestamp_str": data.get('timestamp_str', ''),
                        "top_k": data.get('top_k', 0),
                        "file_size": stat.st_size,
                        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
                except Exception as e:
                    # Skip files that can't be read
                    continue
        
        # Sort by timestamp (newest first)
        search_files.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return jsonify({
            "status": "success",
            "searches": search_files,
            "count": len(search_files)
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/api/status/<job_id>', methods=['GET'])
def get_status(job_id: str):
    """Get job status and logs"""
    if job_id not in jobs:
        return jsonify({"status": "error", "message": "Job not found"}), 404
    
    logger = jobs[job_id]
    return jsonify({
        "job_id": job_id,
        "status": logger.status,
        "logs": logger.get_logs()
    })


@app.route('/api/results/<job_id>', methods=['GET'])
def get_results(job_id: str):
    """Get final results"""
    if job_id not in jobs:
        return jsonify({"status": "error", "message": "Job not found"}), 404
    
    logger = jobs[job_id]
    
    if logger.status != "completed":
        return jsonify({
            "status": logger.status,
            "message": "Job not completed yet"
        }), 202
    
    return jsonify({
        "status": "success",
        "job_id": job_id,
        "results": logger.results
    })


@app.route('/api/logs/<job_id>', methods=['GET'])
def stream_logs(job_id: str):
    """Stream logs in real-time (SSE)"""
    if job_id not in jobs:
        return jsonify({"status": "error", "message": "Job not found"}), 404
    
    def generate():
        logger = jobs[job_id]
        last_index = 0
        
        while logger.status in ["pending", "running"]:
            logs = logger.get_logs()
            if len(logs) > last_index:
                for log in logs[last_index:]:
                    yield f"data: {json.dumps(log)}\n\n"
                last_index = len(logs)
            
            import time
            time.sleep(0.5)
        
        # Send final logs
        logs = logger.get_logs()
        if len(logs) > last_index:
            for log in logs[last_index:]:
                yield f"data: {json.dumps(log)}\n\n"
        
        yield f"data: {json.dumps({'status': 'completed'})}\n\n"
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')


@app.route('/api/integrate', methods=['POST'])
def integrate():
    """Table integration endpoint - integrates tables from search results"""
    try:
        data = request.json or {}
        job_id = data.get('job_id')
        search_type = data.get('search_type', 'single_column')  # Which search type to use
        integration_type = data.get('integration_type', 'union')  # union or intersection
        k = data.get('k', 10)  # Number of tables to integrate, and max rows in result
        
        if not job_id:
            return jsonify({"status": "error", "message": "job_id is required"}), 400
        
        # Find the search results file
        # Try job_id-based filename first (for backward compatibility)
        search_results_path = os.path.join('data', f'compare_search_{job_id}.json')
        
        # If not found, try to find by job_id in saved results
        if not os.path.exists(search_results_path):
            # Look for files containing this job_id
            data_dir = 'data'
            if os.path.exists(data_dir):
                for filename in os.listdir(data_dir):
                    if filename.startswith('search_') and filename.endswith('.json'):
                        file_path = os.path.join(data_dir, filename)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            if data.get('job_id') == job_id:
                                search_results_path = file_path
                                break
                        except:
                            continue
        
        if not os.path.exists(search_results_path):
            return jsonify({
                "status": "error",
                "message": f"Search results not found for job_id: {job_id}. Please run a search first."
            }), 404
        
        # Run integration
        result = integrate_tables_from_search_results(
            search_results_path,
            search_type=search_type,
            integration_type=integration_type,
            k=k
        )
        
        if not result["success"]:
            return jsonify({
                "status": "error",
                "message": result.get("error", "Integration failed")
            }), 500
        
        # Convert DataFrame to JSON-serializable format
        integrated_df = result["integrated_table"]
        integrated_data = {
            "columns": list(integrated_df.columns),
            "data": integrated_df.fillna("").astype(str).values.tolist(),
            "shape": list(integrated_df.shape)
        }
        
        # Save integration results
        integration_output_path = os.path.join('data', f'integration_{job_id}_{search_type}.json')
        os.makedirs('data', exist_ok=True)
        integration_result = {
            "job_id": job_id,
            "search_type": search_type,
            "integration_type": integration_type,
            "k": k,
            "stats": result["stats"],
            "integrated_table": integrated_data,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(integration_output_path, 'w', encoding='utf-8') as f:
            json.dump(integration_result, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            "status": "success",
            "job_id": job_id,
            "integration_type": integration_type,
            "stats": result["stats"],
            "integrated_table": integrated_data,
            "output_path": integration_output_path
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500


if __name__ == '__main__':
    print("Starting ModelSearch Backend API...")
    print("Endpoints:")
    print("  POST /api/search - Start search pipeline")
    print("  GET /api/status/<job_id> - Get job status and logs")
    print("  GET /api/results/<job_id> - Get final results")
    print("  GET /api/logs/<job_id> - Stream logs (SSE)")
    print("\nAll results saved to data/ directory")
    app.run(host='0.0.0.0', port=5000, debug=True)
