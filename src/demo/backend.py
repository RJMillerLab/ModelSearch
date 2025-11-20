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
        
        # Prepare queries for all three search types
        # Follow Blend_internal logic from README examples:
        # - keyword: use headers (column names) - rowid=-1 in index
        # - single_column: use values from first column (Seekers.SC(dataset[clm_name], k) uses single column)
        # - unionable: use entire DataFrame
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
        else:
            queries_parsed['single_column'] = None
            queries_parsed['keyword'] = None
            queries_parsed['unionable'] = None
        
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
                'unionable': []
            }
        else:
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Submit all tasks
                futures = {}
                
                # Submit Card2Card
                futures['card2card'] = executor.submit(run_card2card)
                
                # Submit all three Card2Tab2Card search types
                for search_type_name in ['single_column', 'keyword', 'unionable']:
                    query_parsed = queries_parsed[search_type_name]
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
        
        # Save results
        output_json = os.path.join('data', f'compare_search_{job_id}.json')
        os.makedirs('data', exist_ok=True)
        results_data = {
            "job_id": job_id,
            "query": query,
            "model_id": model_id,
            "model_url": f"https://huggingface.co/{model_id}",
            "top_k": top_k,
            "card2card_results": card2card_results_with_links,
            "card2tab2card_results": card2tab2card_all_with_links,
            "comparison": comparison,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        
        logger.log(f"✅ Results saved to {output_json}")
        
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
    """Get preview of CSV table (first 5 rows, first 5 columns) from modellake.db"""
    table_path = request.args.get('path')
    if not table_path:
        return jsonify({"status": "error", "message": "path parameter is required"}), 400
    
    # Extract basename from path
    basename = os.path.basename(table_path)
    
    # Try to find CSV path from modellake.db first
    csv_path = None
    try:
        import duckdb
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
                # Map table_group/table_type to directory
                # Based on create_index_duckdb.py, table_group is the basename without extension
                # table_type can be 'ori', 't', etc.
                
                # Try to construct path based on table_type and common patterns
                # table_type 'ori' might be in deduped directories, 't' might be in tables_output
                if table_type == 'ori' or table_type is None:
                    # Try deduped directories first
                    for base_dir in [
                        "data_citationlake/processed/deduped_hugging_csvs",
                        "data_citationlake/processed/deduped_github_csvs"
                    ]:
                        full_path = os.path.join(base_dir, basename)
                        if os.path.exists(full_path):
                            csv_path = full_path
                            break
                
                # If not found, try tables_output
                if not csv_path:
                    full_path = os.path.join("data_citationlake/processed/tables_output", basename)
                    if os.path.exists(full_path):
                        csv_path = full_path
        finally:
            con.close()
    except Exception as e:
        # If DB query fails, fall back to file system search
        pass
    
    # Fallback: Try common locations if not found from DB
    if not csv_path:
        if os.path.exists(table_path):
            csv_path = table_path
        else:
            # Try common locations
            for base_dir in [
                "data_citationlake/processed/deduped_hugging_csvs",
                "data_citationlake/processed/deduped_github_csvs",
                "data_citationlake/processed/tables_output"
            ]:
                full_path = os.path.join(base_dir, basename)
                if os.path.exists(full_path):
                    csv_path = full_path
                    break
    
    if not csv_path or not os.path.exists(csv_path):
        return jsonify({
            "status": "error",
            "message": f"CSV file not found: {basename}. Searched in modellake.db and common directories."
        }), 404
    
    try:
        import pandas as pd
        # Read first 5 rows and first 5 columns
        df = pd.read_csv(csv_path, nrows=5)
        # Limit to first 5 columns
        df_preview = df.iloc[:, :5]
        
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
        
        return jsonify({
            "status": "success",
            "table_path": csv_path,
            "rows": len(df_preview),
            "columns": len(df_preview.columns),
            "html": html_table
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error reading CSV: {str(e)}"
        }), 500


@app.route('/api/search', methods=['POST'])
def search():
    """Main search endpoint - starts pipeline"""
    try:
        data = request.json or {}
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


if __name__ == '__main__':
    print("Starting ModelSearch Backend API...")
    print("Endpoints:")
    print("  POST /api/search - Start search pipeline")
    print("  GET /api/status/<job_id> - Get job status and logs")
    print("  GET /api/results/<job_id> - Get final results")
    print("  GET /api/logs/<job_id> - Stream logs (SSE)")
    print("\nAll results saved to data/ directory")
    app.run(host='0.0.0.0', port=5000, debug=True)
