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
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
print(f"🔍 Backend startup: Loading environment variables...")
print(f"   OPENAI_API_KEY loaded: {os.getenv('OPENAI_API_KEY') is not None}")
if os.getenv('OPENAI_API_KEY'):
    api_key = os.getenv('OPENAI_API_KEY')
    print(f"   API key length: {len(api_key)}")
    print(f"   API key prefix: {api_key[:10]}...")
else:
    print(f"   ⚠️  OPENAI_API_KEY not found in environment")

# Auto-detect device: use CUDA if available, otherwise CPU
def get_device():
    """Auto-detect device: CUDA if available, otherwise CPU"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except:
        pass
    return "cpu"

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.search import (
    search_query2modelcard,
    search_card2card,
    search_card2tab2card,
    get_tables_for_model,
    load_relationship_parquet
)
from src.integration.table_integration import integrate_tables_from_search_results, integrate_tables_from_model_search_results
from src.evaluation.llm import evaluate_diversity_with_llm
from src.qa.llm import answer_question_with_llm


app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Default paths
DEFAULT_EMB_NPZ = "data/card2card_embeddings.npz"
DEFAULT_FAISS_INDEX = "data/card2card.faiss"
DEFAULT_JSONL = "data/card2card_corpus.jsonl"
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


def run_search_pipeline(job_id: str, query: Optional[str] = None, top_k: int = 20, model_id: Optional[str] = None, table_search_k: Optional[int] = None, tab2tab_mode: str = 'search', tab2tab_json: Optional[str] = None, card2card_retrieval_mode: str = 'dense'):
    """Run the complete search pipeline in background
    
    Args:
        job_id: Job identifier
        query: Text query (for query mode)
        top_k: Number of results to return
        model_id: Model ID (for modelid mode)
        table_search_k: Number of tables to retrieve in Card2Tab2Card search (defaults to top_k * 2)
        tab2tab_mode: Mode for tab2tab step - 'search' (temporary search) or 'load' (load from JSON)
        tab2tab_json: JSON string containing saved tab2tab results (required when tab2tab_mode='load')
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
                device=get_device(),
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
        
        def run_card2card_all_modes():
            """Run Card2Card pipeline for all retrieval modes"""
            all_results = {}
            retrieval_modes = ["dense", "sparse", "hybrid"]
            
            for mode in retrieval_modes:
                retrieval_mode_display = {
                    "sparse": "sparse (BM25)",
                    "dense": "dense (FAISS)",
                    "hybrid": "hybrid (BM25 + FAISS)"
                }.get(mode, mode)
                logger.log(f"  [Card2Card-{mode.upper()}] Starting {retrieval_mode_display} semantic search...")
                try:
                    # Increase topk slightly to get more results (e.g., topk=3 -> 5-10 results)
                    # Use max(top_k * 2, top_k + 5) to ensure we get more results
                    expanded_topk = max(top_k * 2, top_k + 5)
                    logger.log(f"  ℹ️  [Card2Card-{mode.upper()}] Using expanded top_k: {expanded_topk} (requested: {top_k})")
                    results = search_card2card(
                        model_id=model_id,
                        emb_npz=DEFAULT_EMB_NPZ,
                        faiss_index=DEFAULT_FAISS_INDEX,
                        top_k=expanded_topk,
                        output_json=None,
                        retrieval_mode=mode,
                        jsonl_path=DEFAULT_JSONL,
                        hybrid_method="rrf"  # Default to RRF for hybrid
                    )
                    # Limit to requested top_k for final results
                    final_results = results[:top_k]
                    all_results[mode] = final_results
                    logger.log(f"  ✅ [Card2Card-{mode.upper()}] Found {len(results)} results (returning top {len(final_results)})")
                except Exception as e:
                    logger.log(f"  ❌ [Card2Card-{mode.upper()}] Error: {str(e)}")
                    import traceback
                    logger.log(f"  Traceback: {traceback.format_exc()}")
                    all_results[mode] = {"error": str(e)}
            
            # Return the selected mode's results as primary, plus all modes
            primary_results = all_results.get(card2card_retrieval_mode, all_results.get("dense", []))
            if isinstance(primary_results, dict) and "error" in primary_results:
                # If primary failed, try to use dense as fallback
                primary_results = all_results.get("dense", [])
            
            return {
                "primary": primary_results if not isinstance(primary_results, dict) else [],
                "all_modes": all_results
            }
        
        def run_card2card():
            """Run Card2Card pipeline (legacy single mode)"""
            return run_card2card_all_modes()["primary"]
        
        def run_card2tab2card_search(search_type_name, query_parsed, table_search_k=None):
            """Run one Card2Tab2Card search type"""
            logger.log(f"  [Card2Tab2Card-{search_type_name}] Starting...")
            try:
                # query_csv is available in the outer scope
                # Define hardcoded JSON file paths to check (in priority order)
                # These are from starmie, Blend_internal, and CitationLake
                base_paths = [
                    ".",  # Current directory
                    "data",
                    "data/tab2tab_cache",
                    "../starmie_internal",
                    "../starmie_internal/data",
                    "../starmie_internal/output",
                    "src/Blend_internal",
                    "../Blend_internal",
                    "../CitationLake",
                    "../CitationLake/data",
                    "../CitationLake/output",
                    "data_citationlake",
                ]
                
                # Map search_type to possible JSON filenames
                json_filename_map = {
                    'keyword': [
                        'keyword_ori_tr_str.json',
                        'keyword_ori_tr_str_proc.json',
                        'keyword_ori_tr.json',
                        'keyword_ori_str.json',
                        'baseline_keyword.json',
                        'baseline_keyword_simple.json',
                        'keyword.json',
                        f'tab2tab_{search_type_name}.json',
                    ],
                    'single_column': [
                        'singcol_joinable_ori_tr_str.json',
                        'singcol_joinable_ori_tr.json',
                        'singcol_joinable_ori_str.json',
                        'baseline_singcol_joinable.json',
                        'baseline_singcol_joinable_withheader.json',
                        'singlecol_joinable.json',
                        'single_column.json',
                        f'tab2tab_{search_type_name}.json',
                    ],
                    'multi_column': [
                        'multicol_joinable.json',
                        'multi_column.json',
                        f'tab2tab_{search_type_name}.json',
                    ],
                    'unionable': [
                        'unionable.json',
                        f'tab2tab_{search_type_name}.json',
                    ],
                    'complex': [
                        'complex.json',
                        f'tab2tab_{search_type_name}.json',
                    ],
                    'correlation': [
                        'correlation.json',
                        f'tab2tab_{search_type_name}.json',
                    ],
                    'imputation': [
                        'imputation.json',
                        f'tab2tab_{search_type_name}.json',
                    ],
                    'augmentation': [
                        'augmentation.json',
                        f'tab2tab_{search_type_name}.json',
                    ],
                    'dependent_data': [
                        'dependent_data.json',
                        f'tab2tab_{search_type_name}.json',
                    ],
                    'feature_for_ml': [
                        'feature_for_ml.json',
                        f'tab2tab_{search_type_name}.json',
                    ],
                    'multi_column_collinearity': [
                        'multi_column_collinearity.json',
                        f'tab2tab_{search_type_name}.json',
                    ],
                    'negative_example': [
                        'negative_example.json',
                        f'tab2tab_{search_type_name}.json',
                    ],
                }
                
                # Get possible filenames for this search_type
                possible_filenames = json_filename_map.get(search_type_name, [f'tab2tab_{search_type_name}.json'])
                
                # Build all possible paths to check
                # Get project root (assuming backend.py is in src/demo/)
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
                
                possible_paths = []
                for base_path in base_paths:
                    for filename in possible_filenames:
                        # Resolve relative paths to project root
                        if os.path.isabs(base_path):
                            full_path = os.path.join(base_path, filename)
                        else:
                            # Relative path - resolve from project root
                            full_path = os.path.join(project_root, base_path, filename)
                        # Normalize path (resolve .. and .)
                        full_path = os.path.normpath(full_path)
                        possible_paths.append(full_path)
                
                # Also add cache file as fallback
                cache_dir = "data/tab2tab_cache"
                os.makedirs(cache_dir, exist_ok=True)
                cache_file = os.path.join(cache_dir, f"tab2tab_{search_type_name}.json")
                if cache_file not in possible_paths:
                    possible_paths.append(cache_file)
                
                # Print all paths being checked
                logger.log(f"  📂 [Card2Tab2Card-{search_type_name}] Checking {len(possible_paths)} possible JSON file paths:")
                for i, path in enumerate(possible_paths, 1):
                    exists = "✅ EXISTS" if os.path.exists(path) else "❌ not found"
                    logger.log(f"     {i}. {path} - {exists}")
                
                # Try to find and load from cache
                cached_results = None
                found_path = None
                
                for path in possible_paths:
                    if os.path.exists(path):
                        found_path = path
                        logger.log(f"  ✅ [Card2Tab2Card-{search_type_name}] Found JSON file: {path}")
                        break
                
                if found_path:
                    try:
                        import json
                        logger.log(f"  📖 [Card2Tab2Card-{search_type_name}] Reading JSON file: {found_path}")
                        with open(found_path, 'r', encoding='utf-8') as f:
                            json_data = json.load(f)
                        
                        # Check JSON format: could be {basename: [basenames]} or {"results": [table_ids]}
                        is_basename_format = False
                        is_table_id_format = False
                        
                        if isinstance(json_data, dict):
                            # Check if it's basename format: {csv_basename: [retrieved_csv_basenames]}
                            # Keys should be strings (basenames), values should be lists
                            sample_keys = list(json_data.keys())[:3] if json_data else []
                            if sample_keys and all(isinstance(k, str) for k in sample_keys):
                                sample_values = [json_data[k] for k in sample_keys if k in json_data]
                                if sample_values and all(isinstance(v, list) for v in sample_values):
                                    is_basename_format = True
                                    logger.log(f"  ℹ️  [Card2Tab2Card-{search_type_name}] Detected basename format JSON: {len(json_data)} query tables")
                            
                            # Check if it's table_id format: {"results": [table_ids], "search_type": ...}
                            if "results" in json_data and isinstance(json_data.get("results"), list):
                                is_table_id_format = True
                                logger.log(f"  ℹ️  [Card2Tab2Card-{search_type_name}] Detected table_id format JSON")
                        
                        if not is_basename_format and not is_table_id_format:
                            logger.log(f"  ⚠️  [Card2Tab2Card-{search_type_name}] Unknown JSON format, will re-search")
                            cached_results = None
                        elif is_basename_format:
                            # Use basename format directly
                            cached_results = {"format": "basename", "data": json_data}
                        elif is_table_id_format:
                            # Verify the cached results match the search_type
                            cached_search_type = json_data.get('search_type', '')
                            if cached_search_type and cached_search_type != search_type_name:
                                logger.log(f"  ⚠️  [Card2Tab2Card-{search_type_name}] JSON search_type ({cached_search_type}) doesn't match current ({search_type_name}), will re-search")
                                cached_results = None
                            else:
                                # Verify table IDs exist in database
                                table_ids = json_data.get('results', [])
                                if table_ids:
                                    import duckdb
                                    con = duckdb.connect(DEFAULT_DB_PATH, read_only=True)
                                    try:
                                        table_ids_str = ','.join(str(tid) for tid in table_ids)
                                        verify_query = f"""
                                            SELECT DISTINCT tableid 
                                            FROM modellake_index 
                                            WHERE tableid IN ({table_ids_str}) AND rowid = -1
                                        """
                                        verified_ids = [row[0] for row in con.execute(verify_query).fetchall()]
                                        
                                        if len(verified_ids) == len(table_ids):
                                            logger.log(f"  ✅ [Card2Tab2Card-{search_type_name}] Cached JSON is valid ({len(table_ids)} table IDs verified)")
                                            cached_results = {"format": "table_id", "results": table_ids}
                                        else:
                                            logger.log(f"  ⚠️  [Card2Tab2Card-{search_type_name}] Cached JSON has invalid table IDs ({len(verified_ids)}/{len(table_ids)} valid), will re-search")
                                            logger.log(f"     Missing table IDs: {set(table_ids) - set(verified_ids)}")
                                            cached_results = None
                                    except Exception as e:
                                        logger.log(f"  ⚠️  [Card2Tab2Card-{search_type_name}] Error verifying cached JSON: {str(e)}, will re-search")
                                        cached_results = None
                                    finally:
                                        con.close()
                                else:
                                    logger.log(f"  ⚠️  [Card2Tab2Card-{search_type_name}] Cached JSON has no table IDs, will re-search")
                                    cached_results = None
                    except json.JSONDecodeError as e:
                        logger.log(f"  ⚠️  [Card2Tab2Card-{search_type_name}] Cached JSON is invalid (JSON decode error: {str(e)}), will re-search")
                        cached_results = None
                    except Exception as e:
                        logger.log(f"  ⚠️  [Card2Tab2Card-{search_type_name}] Error reading cached JSON: {str(e)}, will re-search")
                        cached_results = None
                else:
                    logger.log(f"  ℹ️  [Card2Tab2Card-{search_type_name}] No cached JSON file found: {cache_file}, will perform new search")
                
                # If we have valid cached results, use them
                if cached_results:
                    # Get query tables for the model
                    from src.search.card2tab2card import get_tables_for_model
                    query_tables = get_tables_for_model(
                        model_id=model_id,
                        schema_log_path=DEFAULT_SCHEMA_LOG,
                        use_citationlake=True
                    )
                    
                    # If model has no tables, try to use the query_csv basename
                    if not query_tables and query_csv:
                        query_tables = [query_csv]
                        logger.log(f"  ℹ️  [Card2Tab2Card-{search_type_name}] Model has no tables, using query CSV: {query_csv}")
                    
                    if not query_tables:
                        logger.log(f"  ⚠️  [Card2Tab2Card-{search_type_name}] No query tables found for model {model_id} and no query CSV available")
                        return search_type_name, {"error": f"No query tables found for model {model_id}", "model_ids": [], "intermediate": {}}
                    
                    # Handle different JSON formats
                    if cached_results.get("format") == "basename":
                        # Format: {csv_basename: [retrieved_csv_basenames]}
                        basename_data = cached_results["data"]
                        logger.log(f"  ℹ️  [Card2Tab2Card-{search_type_name}] Using basename format JSON with {len(basename_data)} query tables")
                        
                        # Build table_search_results: use basename as key
                        table_search_results = {}
                        for query_table in query_tables:
                            table_basename = os.path.basename(str(query_table))
                            # Look up in JSON using basename
                            if table_basename in basename_data:
                                retrieved_basenames = basename_data[table_basename]
                                # Ensure it's a list
                                if not isinstance(retrieved_basenames, list):
                                    retrieved_basenames = [retrieved_basenames] if retrieved_basenames else []
                                # Map both full path and basename to retrieved basenames
                                table_search_results[query_table] = retrieved_basenames
                                table_search_results[table_basename] = retrieved_basenames
                                logger.log(f"     Found {len(retrieved_basenames)} results for {table_basename}")
                            else:
                                logger.log(f"     ⚠️  No results in JSON for {table_basename}")
                        
                        if not table_search_results:
                            logger.log(f"  ⚠️  [Card2Tab2Card-{search_type_name}] No matching query tables found in JSON, will re-search")
                            cached_results = None
                        else:
                            logger.log(f"  ℹ️  [Card2Tab2Card-{search_type_name}] Using search_card2tab2card_from_tables with {len(table_search_results)} query tables from basename JSON")
                    else:
                        # Format: {"results": [table_ids]}
                        table_ids = cached_results.get('results', [])
                        logger.log(f"  ℹ️  [Card2Tab2Card-{search_type_name}] Loading from table_id format JSON: {len(table_ids)} table IDs")
                        
                        # Convert table IDs to filenames using database
                        import duckdb
                        con = duckdb.connect(DEFAULT_DB_PATH, read_only=True)
                        try:
                            table_ids_str = ','.join(str(tid) for tid in table_ids)
                            filename_query = f"""
                                SELECT DISTINCT tableid, filename 
                                FROM modellake_index 
                                WHERE tableid IN ({table_ids_str}) AND rowid = -1
                            """
                            filename_results = con.execute(filename_query).fetchall()
                            tableid_to_filename = {tid: filename for tid, filename in filename_results}
                            retrieved_filenames = list(tableid_to_filename.values())
                            
                            logger.log(f"  ℹ️  [Card2Tab2Card-{search_type_name}] Converted {len(retrieved_filenames)} table IDs to filenames")
                            
                            if not retrieved_filenames:
                                logger.log(f"  ⚠️  [Card2Tab2Card-{search_type_name}] No filenames found for table IDs, will re-search")
                                cached_results = None  # Force re-search
                            else:
                                # Build table_search_results dict: map query table to retrieved filenames
                                table_search_results = {}
                                for query_table in query_tables:
                                    table_key = query_table
                                    table_basename = os.path.basename(str(query_table))
                                    table_search_results[table_key] = retrieved_filenames
                                    table_search_results[table_basename] = retrieved_filenames
                                
                                logger.log(f"  ℹ️  [Card2Tab2Card-{search_type_name}] Using search_card2tab2card_from_tables with {len(query_tables)} query tables and {len(retrieved_filenames)} retrieved tables")
                        finally:
                            con.close()
                    
                    # If we have valid table_search_results, proceed with search
                    if cached_results and 'table_search_results' in locals() and table_search_results:
                            
                            # Use search_card2tab2card_from_tables
                            from src.search.card2tab2card import search_card2tab2card_from_tables
                            results = search_card2tab2card_from_tables(
                                model_id=model_id,
                                table_search_results=table_search_results,
                                relationship_parquet=DEFAULT_RELATIONSHIP_PARQUET,
                                schema_log_path=DEFAULT_SCHEMA_LOG,
                                use_citationlake=True,
                                k=top_k,
                                modelcard_k=top_k
                            )
                            
                            # Build intermediate data structure
                            intermediate_data = {
                                "retrieved_table_ids": table_ids,
                                "retrieved_table_filenames": retrieved_filenames,
                                "table_id_to_filename": {str(tid): filename for tid, filename in tableid_to_filename.items()},
                                "table_to_models": {}  # Will be populated by search_card2tab2card_from_tables if needed
                            }
                            
                            logger.log(f"  ✅ [Card2Tab2Card-{search_type_name}] Found {len(results)} results from cached JSON")
                            return search_type_name, {
                                "model_ids": results,
                                "intermediate": intermediate_data
                            }
                    else:
                        # No valid results from cached JSON
                        cached_results = None
                
                # If we reach here, we need to perform a new search
                # Check if we should load from manually provided JSON (old behavior)
                if tab2tab_mode == 'load' and tab2tab_json:
                    logger.log(f"  ℹ️  [Card2Tab2Card-{search_type_name}] Loading from saved JSON results...")
                    try:
                        import json
                        saved_results = json.loads(tab2tab_json)
                        
                        # Check if saved results match the current search type
                        saved_search_type = saved_results.get('search_type', '')
                        if saved_search_type and saved_search_type != search_type_name:
                            logger.log(f"  ⚠️  [Card2Tab2Card-{search_type_name}] Saved JSON search_type ({saved_search_type}) doesn't match current search_type ({search_type_name}), using anyway...")
                        
                        # Extract table IDs from saved results
                        # Format from tab2tab.py: {"query": ..., "search_type": ..., "k": ..., "results": [table_ids], ...}
                        table_ids = saved_results.get('results', [])
                        if not table_ids:
                            logger.log(f"  ⚠️  [Card2Tab2Card-{search_type_name}] No table IDs found in saved JSON")
                            return search_type_name, {"error": "No table IDs in saved JSON", "model_ids": [], "intermediate": {}}
                        
                        logger.log(f"  ℹ️  [Card2Tab2Card-{search_type_name}] Found {len(table_ids)} table IDs in saved JSON (saved search_type: {saved_search_type})")
                        
                        # Convert table IDs to filenames using database
                        import duckdb
                        con = duckdb.connect(DEFAULT_DB_PATH, read_only=True)
                        try:
                            table_ids_str = ','.join(str(tid) for tid in table_ids)
                            filename_query = f"""
                                SELECT DISTINCT tableid, filename 
                                FROM modellake_index 
                                WHERE tableid IN ({table_ids_str}) AND rowid = -1
                            """
                            filename_results = con.execute(filename_query).fetchall()
                            tableid_to_filename = {tid: filename for tid, filename in filename_results}
                            retrieved_filenames = list(tableid_to_filename.values())
                            
                            logger.log(f"  ℹ️  [Card2Tab2Card-{search_type_name}] Converted {len(retrieved_filenames)} table IDs to filenames")
                            
                            if not retrieved_filenames:
                                logger.log(f"  ⚠️  [Card2Tab2Card-{search_type_name}] No filenames found for table IDs")
                                return search_type_name, {"error": "No filenames found for table IDs", "model_ids": [], "intermediate": {}}
                            
                            # Get query tables for the model
                            from src.search.card2tab2card import get_tables_for_model
                            query_tables = get_tables_for_model(
                                model_id=model_id,
                                schema_log_path=DEFAULT_SCHEMA_LOG,
                                use_citationlake=True
                            )
                            
                            if not query_tables:
                                logger.log(f"  ⚠️  [Card2Tab2Card-{search_type_name}] No query tables found for model {model_id}")
                                return search_type_name, {"error": f"No query tables found for model {model_id}", "model_ids": [], "intermediate": {}}
                            
                            # Build table_search_results dict: map query table to retrieved filenames
                            # For simplicity, map all query tables to all retrieved filenames
                            # (In a more sophisticated implementation, we could match based on search_type)
                            table_search_results = {}
                            for query_table in query_tables:
                                table_key = query_table
                                table_basename = os.path.basename(str(query_table))
                                table_search_results[table_key] = retrieved_filenames
                                table_search_results[table_basename] = retrieved_filenames
                            
                            logger.log(f"  ℹ️  [Card2Tab2Card-{search_type_name}] Using search_card2tab2card_from_tables with {len(query_tables)} query tables and {len(retrieved_filenames)} retrieved tables")
                            
                            # Use search_card2tab2card_from_tables
                            from src.search.card2tab2card import search_card2tab2card_from_tables
                            results = search_card2tab2card_from_tables(
                                model_id=model_id,
                                table_search_results=table_search_results,
                                relationship_parquet=DEFAULT_RELATIONSHIP_PARQUET,
                                schema_log_path=DEFAULT_SCHEMA_LOG,
                                use_citationlake=True,
                                k=top_k,
                                modelcard_k=top_k
                            )
                            
                            # Build intermediate data structure
                            if cached_results.get("format") == "basename":
                                # For basename format, collect all retrieved basenames
                                all_retrieved = set()
                                for fnames in table_search_results.values():
                                    all_retrieved.update(fnames)
                                intermediate_data = {
                                    "retrieved_table_ids": [],
                                    "retrieved_table_filenames": list(all_retrieved),
                                    "table_id_to_filename": {},
                                    "table_to_models": {},
                                    "source": "basename_json"
                                }
                            else:
                                # For table_id format
                                intermediate_data = {
                                    "retrieved_table_ids": table_ids,
                                    "retrieved_table_filenames": retrieved_filenames,
                                    "table_id_to_filename": {str(tid): filename for tid, filename in tableid_to_filename.items()},
                                    "table_to_models": {},
                                    "source": "table_id_json"
                                }
                            
                            logger.log(f"  ✅ [Card2Tab2Card-{search_type_name}] Found {len(results)} results from saved JSON")
                            return search_type_name, {
                                "model_ids": results,
                                "intermediate": intermediate_data
                            }
                        finally:
                            con.close()
                    except json.JSONDecodeError as e:
                        logger.log(f"  ❌ [Card2Tab2Card-{search_type_name}] JSON decode error: {str(e)}")
                        return search_type_name, {"error": f"Invalid JSON: {str(e)}"}
                    except Exception as e:
                        logger.log(f"  ❌ [Card2Tab2Card-{search_type_name}] Error loading from JSON: {str(e)}")
                        import traceback
                        logger.log(f"  Traceback: {traceback.format_exc()}")
                        return search_type_name, {"error": str(e)}
                
                # Default: temporary dataset search
                # Use a temporary JSON file to capture intermediate results
                import tempfile
                import json
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                    tmp_json_path = tmp_file.name
                
                # Validate query_parsed before proceeding
                if query_parsed is None:
                    logger.log(f"  ⚠️  [Card2Tab2Card-{search_type_name}] query_parsed is None, cannot perform search")
                    return search_type_name, {"error": f"query_parsed is None for {search_type_name}", "model_ids": [], "intermediate": {}}
                
                # Log query details for debugging
                if search_type_name == 'single_column':
                    logger.log(f"  ℹ️  [Card2Tab2Card-{search_type_name}] Query: {len(query_parsed) if query_parsed else 0} values")
                    if query_parsed:
                        logger.log(f"     Sample values: {query_parsed[:3]}{'...' if len(query_parsed) > 3 else ''}")
                elif search_type_name == 'keyword':
                    logger.log(f"  ℹ️  [Card2Tab2Card-{search_type_name}] Query: {len(query_parsed) if query_parsed else 0} keywords")
                    if query_parsed:
                        logger.log(f"     Keywords: {query_parsed[:5]}{'...' if len(query_parsed) > 5 else ''}")
                elif search_type_name in ['multi_column', 'unionable', 'complex', 'correlation', 'imputation', 'augmentation', 'dependent_data', 'feature_for_ml', 'multi_column_collinearity', 'negative_example']:
                    if query_parsed is not None:
                        logger.log(f"  ℹ️  [Card2Tab2Card-{search_type_name}] Query: DataFrame with {len(query_parsed)} rows, {len(query_parsed.columns) if hasattr(query_parsed, 'columns') else 'N/A'} columns")
                    else:
                        logger.log(f"  ⚠️  [Card2Tab2Card-{search_type_name}] Query: DataFrame is None")
                
                # Use provided table_search_k or default to top_k * 1.5 (minimum 20, max 20) for table search
                # This allows more tables to be retrieved, then we limit modelcards to top_k
                if table_search_k is None:
                    table_search_k = min(max(int(top_k * 1.5), 20), 20)  # Get more tables, then filter to top_k modelcards (max 20)
                
                logger.log(f"  ℹ️  [Card2Tab2Card-{search_type_name}] Table search k: {table_search_k}, ModelCard k: {top_k}")
                logger.log(f"  ℹ️  [Card2Tab2Card-{search_type_name}] DB path: {DEFAULT_DB_PATH}")
                logger.log(f"  ℹ️  [Card2Tab2Card-{search_type_name}] DB exists: {os.path.exists(DEFAULT_DB_PATH) if DEFAULT_DB_PATH else False}")
                
                logger.log(f"  🔍 [Card2Tab2Card-{search_type_name}] Performing new tab2tab search using Blend_internal...")
                try:
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
                    logger.log(f"  ✅ [Card2Tab2Card-{search_type_name}] Search completed, got {len(results) if results else 0} results")
                except Exception as e:
                    logger.log(f"  ❌ [Card2Tab2Card-{search_type_name}] Error during search: {str(e)}")
                    import traceback
                    logger.log(f"  Traceback: {traceback.format_exc()}")
                    return search_type_name, {"error": f"Search failed: {str(e)}", "model_ids": [], "intermediate": {}}
                
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
                                    
                                    # Save tab2tab results to cache for future use
                                    table_ids = intermediate_data.get('retrieved_table_ids', [])
                                    if table_ids:
                                        try:
                                            cache_data = {
                                                "query": str(query_parsed) if query_parsed is not None else None,
                                                "search_type": search_type_name,
                                                "k": table_search_k,
                                                "results": table_ids,
                                                "num_results": len(table_ids),
                                                "timestamp": datetime.now().isoformat()
                                            }
                                            
                                            # Save to cache file (use the standard cache directory)
                                            with open(cache_file, 'w', encoding='utf-8') as f:
                                                json.dump(cache_data, f, ensure_ascii=False, indent=2)
                                            
                                            logger.log(f"  ✅ [Card2Tab2Card-{search_type_name}] Saved tab2tab results to cache: {os.path.abspath(cache_file)} ({len(table_ids)} table IDs)")
                                        except Exception as e:
                                            logger.log(f"  ⚠️  [Card2Tab2Card-{search_type_name}] Error saving to cache: {str(e)}")
                                            # Don't fail the search if cache save fails
                                    else:
                                        logger.log(f"  ⚠️  [Card2Tab2Card-{search_type_name}] No table IDs in results, skipping cache save")
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
            
            # For multi_column: use entire DataFrame (finds tables with overlapping values across multiple columns)
            queries_parsed['multi_column'] = query_df
            logger.log(f"ℹ️  Multi_column search using DataFrame with {len(query_df)} rows and {len(query_df.columns)} columns")
            
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
            
            # For imputation: use entire DataFrame, will extract examples (complete rows) and queries (missing values)
            queries_parsed['imputation'] = query_df
            logger.log(f"ℹ️  Imputation search using DataFrame with {len(query_df)} rows and {len(query_df.columns)} columns")
            if len(query_df.columns) >= 2:
                complete_rows = query_df[query_df.iloc[:, 1].notna()].shape[0]
                missing_rows = query_df[query_df.iloc[:, 1].isna()].shape[0]
                logger.log(f"   Examples (complete rows): {complete_rows}, Queries (missing rows): {missing_rows}")
            else:
                logger.log(f"   ⚠️  DataFrame needs at least 2 columns for imputation")
            
            # For augmentation: use entire DataFrame, will extract examples (complete rows) and queries (missing values)
            queries_parsed['augmentation'] = query_df
            logger.log(f"ℹ️  Augmentation search using DataFrame with {len(query_df)} rows and {len(query_df.columns)} columns")
            if len(query_df.columns) >= 2:
                complete_rows = query_df[query_df.iloc[:, 1].notna()].shape[0]
                missing_rows = query_df[query_df.iloc[:, 1].isna()].shape[0]
                logger.log(f"   Examples (complete rows): {complete_rows}, Queries (missing rows): {missing_rows}")
            else:
                logger.log(f"   ⚠️  DataFrame needs at least 2 columns for augmentation")
            
            # For dependent_data: use entire DataFrame, will extract two pairs of dependent columns
            queries_parsed['dependent_data'] = query_df
            logger.log(f"ℹ️  Dependent data search using DataFrame with {len(query_df)} rows and {len(query_df.columns)} columns")
            if len(query_df.columns) >= 2:
                logger.log(f"   Will use first 2 columns as first pair, columns 2-3 (or 1-2) as second pair")
            else:
                logger.log(f"   ⚠️  DataFrame needs at least 2 columns for dependent_data")
            
            # For feature_for_ml: use entire DataFrame, will extract source, target, and feature columns
            queries_parsed['feature_for_ml'] = query_df
            logger.log(f"ℹ️  Feature for ML search using DataFrame with {len(query_df)} rows and {len(query_df.columns)} columns")
            if numeric_cols and len(numeric_cols) >= 2:
                logger.log(f"   Will use first column as source, '{numeric_cols[0]}' as target, '{numeric_cols[1]}' as feature")
            else:
                logger.log(f"   ⚠️  DataFrame needs at least 2 numeric columns for feature_for_ml")
            
            # For multi_column_collinearity: use entire DataFrame, will extract columns
            queries_parsed['multi_column_collinearity'] = query_df
            logger.log(f"ℹ️  Multi-column collinearity search using DataFrame with {len(query_df)} rows and {len(query_df.columns)} columns")
            if numeric_cols and len(numeric_cols) >= 2:
                logger.log(f"   Will use first column as source, '{numeric_cols[0]}' as target, '{numeric_cols[1]}' as feature, first 2 columns for multi-column")
            else:
                logger.log(f"   ⚠️  DataFrame needs at least 2 numeric columns for multi_column_collinearity")
            
            # For negative_example: use entire DataFrame, will split into inclusive and exclusive
            queries_parsed['negative_example'] = query_df
            logger.log(f"ℹ️  Negative example search using DataFrame with {len(query_df)} rows and {len(query_df.columns)} columns")
            if len(query_df) >= 2:
                logger.log(f"   Will split DataFrame: first half as inclusive, second half as exclusive")
            else:
                logger.log(f"   ⚠️  DataFrame needs at least 2 rows for negative_example")
        else:
            queries_parsed['single_column'] = None
            queries_parsed['keyword'] = None
            queries_parsed['multi_column'] = None
            queries_parsed['unionable'] = None
            queries_parsed['complex'] = None
            queries_parsed['correlation'] = None
            queries_parsed['imputation'] = None
            queries_parsed['augmentation'] = None
            queries_parsed['dependent_data'] = None
            queries_parsed['feature_for_ml'] = None
            queries_parsed['multi_column_collinearity'] = None
            queries_parsed['negative_example'] = None
        
        # Run all searches in parallel using ThreadPoolExecutor
        card2card_results = None
        card2tab2card_all = {}
        
        # Only run Card2Tab2Card if model has tables
        # Check if we have JSON files available (even if model has no tables)
        has_json_files = False
        if query_df is not None:
            # Check if any JSON files exist for the search types we'll run
            cache_dir = "data/tab2tab_cache"
            base_paths = [
                ".", "data", cache_dir,
                "../starmie_internal", "../starmie_internal/data", "../starmie_internal/output",
                "src/Blend_internal", "../Blend_internal",
                "../CitationLake", "../CitationLake/data", "../CitationLake/output",
                "data_citationlake",
            ]
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
            
            # Check for common JSON filenames
            search_types_to_check = ['keyword', 'single_column', 'multi_column', 'unionable']
            for search_type in search_types_to_check:
                possible_filenames = []
                if search_type == 'keyword':
                    possible_filenames = ['keyword_ori_tr_str.json', 'keyword_ori_tr.json', 'baseline_keyword.json']
                elif search_type == 'single_column':
                    possible_filenames = ['singcol_joinable_ori_tr_str.json', 'baseline_singcol_joinable.json']
                
                for base_path in base_paths:
                    for filename in possible_filenames:
                        if os.path.isabs(base_path):
                            full_path = os.path.join(base_path, filename)
                        else:
                            full_path = os.path.join(project_root, base_path, filename)
                        full_path = os.path.normpath(full_path)
                        if os.path.exists(full_path):
                            has_json_files = True
                            logger.log(f"  ℹ️  Found JSON file for Card2Tab2Card: {full_path}")
                            break
                    if has_json_files:
                        break
                if has_json_files:
                    break
        
        if not has_tables and not has_json_files:
            logger.log("⚠️  Skipping Card2Tab2Card pipeline - model has no tables and no JSON files found")
            # Only run Card2Card (all modes)
            card2card_all_modes_result = run_card2card_all_modes()
            card2card_results = card2card_all_modes_result["primary"]
            card2card_all_modes = card2card_all_modes_result["all_modes"]
            # Set empty results for Card2Tab2Card
            card2tab2card_all = {
                'single_column': [],
                'keyword': [],
                'multi_column': [],
                'unionable': [],
                'complex': [],
                'correlation': [],
                'imputation': [],
                'augmentation': [],
                'dependent_data': [],
                'feature_for_ml': [],
                'multi_column_collinearity': [],
                'negative_example': []
            }
        else:
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Submit all tasks
                futures = {}
                
                # Submit Card2Card (all modes)
                futures['card2card'] = executor.submit(run_card2card_all_modes)
                
                # Submit all Card2Tab2Card search types (run all: single_column, keyword, multi_column, unionable, complex, correlation, imputation, augmentation, dependent_data, feature_for_ml, multi_column_collinearity, negative_example)
                all_search_types = ['single_column', 'keyword', 'multi_column', 'unionable', 'complex', 'correlation', 'imputation', 'augmentation', 'dependent_data', 'feature_for_ml', 'multi_column_collinearity', 'negative_example']
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
                        # Card2Card result (can be dict with all_modes or legacy list)
                        if isinstance(result, dict) and "all_modes" in result:
                            card2card_all_modes = result["all_modes"]
                            card2card_results = result["primary"]
                        else:
                            card2card_results = result
                            # Create all_modes dict for legacy format
                            card2card_all_modes = {
                                card2card_retrieval_mode: card2card_results
                            }
                            for mode in ["dense", "sparse", "hybrid"]:
                                if mode != card2card_retrieval_mode:
                                    card2card_all_modes[mode] = []
                except Exception as e:
                    logger.log(f"  ❌ Future error: {str(e)}")
        
        # Handle card2card results (can be dict with all_modes or list)
        card2card_all_modes = {}
        if isinstance(card2card_results, dict) and "all_modes" in card2card_results:
            # New format with all modes
            card2card_all_modes = card2card_results["all_modes"]
            card2card_results = card2card_results["primary"]
        else:
            # Legacy format - only one mode, create all_modes dict
            card2card_all_modes = {
                card2card_retrieval_mode: card2card_results
            }
            # Fill other modes with empty or error
            for mode in ["dense", "sparse", "hybrid"]:
                if mode != card2card_retrieval_mode:
                    card2card_all_modes[mode] = []
        
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
        # Add hyperlinks to all modes
        card2card_all_modes_with_links = {}
        for mode, results in card2card_all_modes.items():
            if isinstance(results, dict) and "error" in results:
                card2card_all_modes_with_links[mode] = results
            else:
                card2card_all_modes_with_links[mode] = add_hyperlinks(results)
        
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
        
        # Save results to timestamp-based folder
        # Use query time (start_time) for timestamp to ensure consistency
        timestamp_str = datetime.fromtimestamp(start_time).strftime('%Y%m%d_%H%M%S')
        
        # Create folder name based on mode
        if query:
            # Query mode: use first 30 chars of query (sanitized)
            query_safe = "".join(c for c in query[:30] if c.isalnum() or c in (' ', '-', '_')).strip()
            query_safe = query_safe.replace(' ', '_')
            folder_name = f'{timestamp_str}_{query_safe}'
        else:
            # ModelID mode: use model_id (sanitized)
            model_id_safe = model_id.replace('/', '_').replace('\\', '_')
            folder_name = f'{timestamp_str}_{model_id_safe}'
        
        # Limit folder name length
        if len(folder_name) > 150:
            folder_name = folder_name[:150]
        
        # Create folder and save results
        search_folder = os.path.join('data', folder_name)
        os.makedirs(search_folder, exist_ok=True)
        
        # Save main results file
        filename = 'search_results.json'
        output_json = os.path.join(search_folder, filename)
        
        results_data = {
            "job_id": job_id,
            "query": query,
            "model_id": model_id,
            "model_url": f"https://huggingface.co/{model_id}",
            "top_k": top_k,
            "table_search_k": table_search_k,
            "card2card_retrieval_mode": card2card_retrieval_mode,
            "card2card_results": card2card_results_with_links,
            "card2card_all_modes": card2card_all_modes_with_links,
            "card2tab2card_results": card2tab2card_all_with_links,
            "comparison": comparison,
            "timestamp": datetime.fromtimestamp(start_time).isoformat(),
            "timestamp_str": timestamp_str,
            "folder_name": folder_name,
            "filename": filename,
            "folder_path": search_folder
        }
        
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        
        logger.log(f"✅ Results saved to {output_json}")
        logger.log(f"📁 Search folder: {search_folder}")
        
        # Also save with job_id for backward compatibility (in the same folder)
        job_id_json = os.path.join(search_folder, f'compare_search_{job_id}.json')
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
    
    # URL decode the path in case it was encoded
    from urllib.parse import unquote
    table_path = unquote(table_path)
    
    # Extract basename from path (handles both full paths and basenames)
    # If table_path is already just a basename, basename will be the same
    basename = os.path.basename(table_path)
    
    # Log the request
    print(f"\n{'='*60}")
    print(f"🔍 Table Preview Request")
    print(f"{'='*60}")
    print(f"Input path (raw): {request.args.get('path')}")
    print(f"Input path (decoded): {table_path}")
    print(f"Basename: {basename}")
    print(f"Is absolute path: {os.path.isabs(table_path)}")
    print(f"Path exists directly: {os.path.exists(table_path)}")
    
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
    
    # Initialize csv_path to None
    csv_path = None
    
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
            folder_name = data.get('folder_name')  # Can be timestamp folder or 'template'
            if not folder_name:
                return jsonify({"status": "error", "message": "folder_name is required for mimic mode"}), 400
            
            # Determine file path
            if folder_name == 'template':
                # Load from template folder
                file_path = os.path.join('data', 'template', 'search_results.json')
            else:
                # Load from timestamp folder
                file_path = os.path.join('data', folder_name, 'search_results.json')
            
            if not os.path.exists(file_path):
                return jsonify({
                    "status": "error",
                    "message": f"Saved results not found: {folder_name}"
                }), 404
            
            # Load the saved results
            with open(file_path, 'r', encoding='utf-8') as f:
                saved_results = json.load(f)
            
            # Create a new job_id for this mimic session
            job_id = str(uuid.uuid4())
            jobs[job_id] = JobLogger(job_id)
            jobs[job_id].set_results(saved_results)
            jobs[job_id].set_status("completed")
            jobs[job_id].log(f"✅ Loaded saved results from {folder_name}")
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
        tab2tab_mode = data.get('tab2tab_mode', 'search')  # 'search' or 'load'
        tab2tab_json = data.get('tab2tab_json', None)  # JSON string when tab2tab_mode='load'
        card2card_retrieval_mode = data.get('card2card_retrieval_mode', 'dense')  # 'sparse', 'dense', or 'hybrid'
        
        # Validate card2card_retrieval_mode
        if card2card_retrieval_mode not in ['sparse', 'dense', 'hybrid']:
            return jsonify({"status": "error", "message": f"Invalid card2card_retrieval_mode: {card2card_retrieval_mode}. Must be 'sparse', 'dense', or 'hybrid'"}), 400
        
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
        
        # Validate tab2tab mode
        if tab2tab_mode == 'load' and not tab2tab_json:
            return jsonify({"status": "error", "message": "tab2tab_json is required when tab2tab_mode='load'"}), 400
        
        # Create job
        job_id = str(uuid.uuid4())
        jobs[job_id] = JobLogger(job_id)
        
        # Start pipeline in background thread
        thread = threading.Thread(
            target=run_search_pipeline,
            args=(job_id, query, top_k, model_id, table_search_k, tab2tab_mode, tab2tab_json, card2card_retrieval_mode)
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


# Path to preset queries JSON (for debug / benchmark); relative to project root (cwd when running backend)
PRESET_QUERIES_PATH = os.path.join("data", "preset_queries.json")


@app.route('/api/preset-queries', methods=['GET'])
def get_preset_queries():
    """Return preset benchmark/debug queries from data/preset_queries.json."""
    try:
        path = PRESET_QUERIES_PATH
        if not os.path.exists(path):
            return jsonify({
                "status": "success",
                "queries": [],
                "message": "Preset queries file not found; use data/preset_queries.json to add presets."
            })
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        queries = data.get("queries", [])
        return jsonify({
            "status": "success",
            "queries": queries,
            "description": data.get("description", "")
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/saved-searches', methods=['GET'])
def list_saved_searches():
    """List all saved search results (organized by timestamp folders)"""
    try:
        data_dir = 'data'
        print(f"\n{'='*60}")
        print(f"🔍 Saved Searches API Request")
        print(f"{'='*60}")
        print(f"Data directory: {os.path.abspath(data_dir)}")
        print(f"Directory exists: {os.path.exists(data_dir)}")
        
        if not os.path.exists(data_dir):
            print("⚠️  Data directory does not exist")
            return jsonify({
                "status": "success",
                "searches": [],
                "template_available": False
            })
        
        # Find all search folders (timestamp-based folders)
        search_folders = []
        
        # Check for template folder
        template_path = os.path.join(data_dir, 'template', 'search_results.json')
        template_available = os.path.exists(template_path)
        print(f"Template available: {template_available}")
        
        # Scan data directory for timestamp folders
        print(f"\nScanning data directory...")
        items = os.listdir(data_dir)
        print(f"Found {len(items)} items in data directory")
        
        for item in items:
            item_path = os.path.join(data_dir, item)
            
            # Skip if not a directory or if it's special folders
            if not os.path.isdir(item_path) or item in ['template', 'benchmarks']:
                if os.path.isdir(item_path):
                    print(f"  ⏭️  Skipping special folder: {item}")
                continue
            
            # Check if folder contains search_results.json
            search_file = os.path.join(item_path, 'search_results.json')
            print(f"  📁 Checking: {item}")
            print(f"     File exists: {os.path.exists(search_file)}")
            
            if os.path.exists(search_file):
                try:
                    # Read metadata without loading full file
                    with open(search_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Get folder stats
                    stat = os.stat(search_file)
                    
                    folder_info = {
                        "folder_name": item,
                        "query": data.get('query', ''),
                        "model_id": data.get('model_id', ''),
                        "timestamp": data.get('timestamp', ''),
                        "timestamp_str": data.get('timestamp_str', ''),
                        "top_k": data.get('top_k', 0),
                        "file_size": stat.st_size,
                        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    }
                    search_folders.append(folder_info)
                    print(f"     ✅ Added: {item} (query={data.get('query')}, model_id={data.get('model_id')})")
                except Exception as e:
                    # Skip folders that can't be read
                    print(f"     ❌ Error reading {item}: {str(e)}")
                    continue
            else:
                print(f"     ⚠️  No search_results.json found")
        
        # Sort by timestamp (newest first)
        search_folders.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        print(f"\n✅ Found {len(search_folders)} saved searches")
        print(f"{'='*60}\n")
        
        return jsonify({
            "status": "success",
            "searches": search_folders,
            "count": len(search_folders),
            "template_available": template_available
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
        integration_type = data.get('integration_type', 'union')  # union, intersection, alite, or outer_join
        k = data.get('k', 10)  # Number of tables to integrate, and max rows in result
        
        if not job_id:
            return jsonify({"status": "error", "message": "job_id is required"}), 400
        
        # Find the search results file
        # Try multiple strategies to find the search results
        search_results_path = None
        
        # Strategy 1: Try job_id-based filename (for backward compatibility)
        search_results_path = os.path.join('data', f'compare_search_{job_id}.json')
        if not os.path.exists(search_results_path):
            # Strategy 2: Look in saved search folders
            data_dir = 'data'
            if os.path.exists(data_dir):
                # Check all subdirectories for search_results.json
                for folder_name in os.listdir(data_dir):
                    folder_path = os.path.join(data_dir, folder_name)
                    if os.path.isdir(folder_path):
                        # Check for search_results.json in this folder
                        potential_path = os.path.join(folder_path, 'search_results.json')
                        if os.path.exists(potential_path):
                            try:
                                with open(potential_path, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                if data.get('job_id') == job_id:
                                    search_results_path = potential_path
                                    print(f"✅ Found search results in folder: {folder_path}")
                                    break
                            except Exception as e:
                                print(f"⚠️  Error reading {potential_path}: {e}")
                                continue
                        
                        # Also check for compare_search_{job_id}.json in folder
                        potential_path = os.path.join(folder_path, f'compare_search_{job_id}.json')
                        if os.path.exists(potential_path):
                            search_results_path = potential_path
                            print(f"✅ Found search results (legacy format) in folder: {folder_path}")
                            break
        
        if not search_results_path or not os.path.exists(search_results_path):
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


@app.route('/api/integrate-model-search', methods=['POST'])
def integrate_model_search():
    """Table integration endpoint for model search (Card2Card) results"""
    try:
        data = request.json or {}
        job_id = data.get('job_id')
        integration_type = data.get('integration_type', 'union')  # union, intersection, alite, or outer_join
        k = data.get('k', 10)  # Number of tables to integrate, and max rows in result
        max_models = data.get('max_models', 10)  # Maximum number of models to process
        
        if not job_id:
            return jsonify({"status": "error", "message": "job_id is required"}), 400
        
        # Find the search results file
        search_results_path = None
        
        # Strategy 1: Try job_id-based filename (for backward compatibility)
        search_results_path = os.path.join('data', f'compare_search_{job_id}.json')
        if not os.path.exists(search_results_path):
            # Strategy 2: Look in saved search folders
            data_dir = 'data'
            if os.path.exists(data_dir):
                # Check all subdirectories for search_results.json
                for folder_name in os.listdir(data_dir):
                    folder_path = os.path.join(data_dir, folder_name)
                    if os.path.isdir(folder_path):
                        # Check for search_results.json in this folder
                        potential_path = os.path.join(folder_path, 'search_results.json')
                        if os.path.exists(potential_path):
                            try:
                                with open(potential_path, 'r', encoding='utf-8') as f:
                                    folder_data = json.load(f)
                                if folder_data.get('job_id') == job_id:
                                    search_results_path = potential_path
                                    print(f"✅ Found search results in folder: {folder_path}")
                                    break
                            except Exception as e:
                                print(f"⚠️  Error reading {potential_path}: {e}")
                                continue
                        
                        # Also check for compare_search_{job_id}.json in folder
                        potential_path = os.path.join(folder_path, f'compare_search_{job_id}.json')
                        if os.path.exists(potential_path):
                            search_results_path = potential_path
                            print(f"✅ Found search results (legacy format) in folder: {folder_path}")
                            break
        
        if not search_results_path or not os.path.exists(search_results_path):
            return jsonify({
                "status": "error",
                "message": f"Search results not found for job_id: {job_id}. Please run a search first."
            }), 404
        
        # Run integration
        result = integrate_tables_from_model_search_results(
            search_results_path,
            integration_type=integration_type,
            k=k,
            max_models=max_models,
            relationship_parquet=DEFAULT_RELATIONSHIP_PARQUET,
            schema_log_path=DEFAULT_SCHEMA_LOG,
            use_citationlake=True
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
        integration_output_path = os.path.join('data', f'integration_model_search_{job_id}.json')
        os.makedirs('data', exist_ok=True)
        integration_result = {
            "job_id": job_id,
            "integration_type": integration_type,
            "k": k,
            "max_models": max_models,
            "stats": result["stats"],
            "integrated_table": integrated_data,
            "model_ids": result.get("model_ids", []),
            "models_with_tables": result.get("models_with_tables", []),
            "models_without_tables": result.get("models_without_tables", []),
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
            "model_ids": result.get("model_ids", []),
            "models_with_tables": result.get("models_with_tables", []),
            "models_without_tables": result.get("models_without_tables", []),
            "output_path": integration_output_path
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    """Evaluate diversity between two integrated tables using LLM"""
    try:
        data = request.json or {}
        job_id = data.get('job_id')
        integration1_job_id = data.get('integration1_job_id')  # Table search integration job_id
        integration2_job_id = data.get('integration2_job_id')  # Model search integration job_id
        integration1_type = data.get('integration1_type', 'single_column')  # Search type for table search
        integration2_type = data.get('integration2_type', 'model_search')
        use_fake = data.get('use_fake', False)  # Use fake response for testing
        fake_response_path = data.get('fake_response_path')  # Optional path to fake response file
        fake_response_content = data.get('fake_response_content')  # Optional fake response content
        
        if not job_id:
            return jsonify({"status": "error", "message": "job_id is required"}), 400
        
        # Find integration results - automatically discover files based on job_id
        integration1_path = None
        integration2_path = None
        
        # Strategy: Scan for integration results files matching job_id
        data_dir = 'data'
        if os.path.exists(data_dir):
            # For table search integration: look for integration_{job_id}_{search_type}.json
            # Try common search types if specific job_id not provided
            if integration1_job_id:
                # Try with provided integration1_type (which should be search_type)
                potential_path = os.path.join(data_dir, f'integration_{integration1_job_id}_{integration1_type}.json')
                if os.path.exists(potential_path):
                    integration1_path = potential_path
            else:
                # Scan for any integration file matching job_id
                # Try common search types
                for search_type in ['single_column', 'multi_column', 'table_search']:
                    potential_path = os.path.join(data_dir, f'integration_{job_id}_{search_type}.json')
                    if os.path.exists(potential_path):
                        integration1_path = potential_path
                        print(f"✅ Found table search integration: {integration1_path}")
                        break
                
                # If not found, scan all files in data_dir
                if not integration1_path:
                    for filename in os.listdir(data_dir):
                        if filename.startswith(f'integration_{job_id}_') and filename.endswith('.json'):
                            # Exclude model search files
                            if not filename.startswith(f'integration_model_search_{job_id}'):
                                potential_path = os.path.join(data_dir, filename)
                                if os.path.exists(potential_path):
                                    integration1_path = potential_path
                                    print(f"✅ Found table search integration (scanned): {integration1_path}")
                                    break
            
            # For model search integration: look for integration_model_search_{job_id}.json
            if integration2_job_id:
                potential_path = os.path.join(data_dir, f'integration_model_search_{integration2_job_id}.json')
                if os.path.exists(potential_path):
                    integration2_path = potential_path
            else:
                integration2_path = os.path.join(data_dir, f'integration_model_search_{job_id}.json')
                if not os.path.exists(integration2_path):
                    integration2_path = None
        
        # Load integration results
        table1_df = None
        table2_df = None
        query = None
        
        # Load table1 (Table Search Integration)
        if integration1_path and os.path.exists(integration1_path):
            try:
                with open(integration1_path, 'r', encoding='utf-8') as f:
                    integration1_data = json.load(f)
                if integration1_data.get('integrated_table'):
                    import pandas as pd
                    table1_data = integration1_data['integrated_table']
                    if 'data' in table1_data and 'columns' in table1_data:
                        table1_df = pd.DataFrame(table1_data['data'], columns=table1_data['columns'])
                        integration1_type_used = integration1_data.get('integration_type', 'unknown')
                        search_type_used = integration1_data.get('search_type', 'unknown')
                        print(f"✅ Loaded table1 (Table Search): {table1_df.shape[0]} rows × {table1_df.shape[1]} columns")
                        print(f"   Integration type used: {integration1_type_used}, Search type: {search_type_used}")
                    else:
                        print(f"⚠️  Table1 data missing 'data' or 'columns' keys")
            except Exception as e:
                print(f"❌ Error loading table1 from {integration1_path}: {e}")
        
        # Load table2 (Model Search Integration)
        if integration2_path and os.path.exists(integration2_path):
            try:
                with open(integration2_path, 'r', encoding='utf-8') as f:
                    integration2_data = json.load(f)
                if integration2_data.get('integrated_table'):
                    import pandas as pd
                    table2_data = integration2_data['integrated_table']
                    if 'data' in table2_data and 'columns' in table2_data:
                        table2_df = pd.DataFrame(table2_data['data'], columns=table2_data['columns'])
                        integration2_type_used = integration2_data.get('integration_type', 'unknown')
                        print(f"✅ Loaded table2 (Model Search): {table2_df.shape[0]} rows × {table2_df.shape[1]} columns")
                        print(f"   Integration type used: {integration2_type_used}")
                    else:
                        print(f"⚠️  Table2 data missing 'data' or 'columns' keys")
            except Exception as e:
                print(f"❌ Error loading table2 from {integration2_path}: {e}")
        
        # Get query from search results
        search_results_path = None
        for folder_name in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, folder_name)
            if os.path.isdir(folder_path):
                potential_path = os.path.join(folder_path, 'search_results.json')
                if os.path.exists(potential_path):
                    try:
                        with open(potential_path, 'r', encoding='utf-8') as f:
                            search_data = json.load(f)
                        if search_data.get('job_id') == job_id:
                            search_results_path = potential_path
                            query = search_data.get('query', 'Model search query')
                            break
                    except:
                        continue
        
        # Provide helpful error messages
        if table1_df is None:
            error_msg = f"Could not find table search integration results for job_id: {job_id}. "
            error_msg += "Please run 'Table Integration (from Table Search)' first with your desired integration type (union, intersection, etc.)."
            if integration1_path:
                error_msg += f" Expected file: {integration1_path}"
            print(f"❌ {error_msg}")
            return jsonify({
                "status": "error",
                "message": error_msg,
                "integration1_path": integration1_path,
                "integration1_exists": os.path.exists(integration1_path) if integration1_path else False
            }), 404
        
        if table2_df is None:
            error_msg = f"Could not find model search integration results for job_id: {job_id}. "
            error_msg += "Please run 'Table Integration (from Model Search)' first with your desired integration type (union, intersection, etc.)."
            if integration2_path:
                error_msg += f" Expected file: {integration2_path}"
            print(f"❌ {error_msg}")
            return jsonify({
                "status": "error",
                "message": error_msg,
                "integration2_path": integration2_path,
                "integration2_exists": os.path.exists(integration2_path) if integration2_path else False
            }), 404
        
        # Run evaluation
        # Note: table1_df is from table search (integration1), table2_df is from model search (integration2)
        # But for display: Left = Model Search, Right = Table Search
        # So we swap them: table1 should be model search, table2 should be table search
        print(f"📊 Running evaluation with use_fake={use_fake}")
        print(f"   Received use_fake from request: {use_fake}")
        print(f"   Table1 (Model Search) shape: {table2_df.shape if table2_df is not None else 'None'}")
        print(f"   Table2 (Table Search) shape: {table1_df.shape if table1_df is not None else 'None'}")
        
        # If use_fake is False, check if OPENAI_API_KEY is available
        # If not available, return error instead of auto-fallback
        if not use_fake:
            # os is already imported at the top of backend.py
            api_key = os.getenv("OPENAI_API_KEY")
            print(f"🔍 Debug: Checking OPENAI_API_KEY...")
            print(f"   API key exists: {api_key is not None}")
            print(f"   API key length: {len(api_key) if api_key else 0}")
            print(f"   API key prefix: {api_key[:10] + '...' if api_key and len(api_key) > 10 else 'N/A'}")
            print(f"   All env vars with 'OPENAI': {[k for k in os.environ.keys() if 'OPENAI' in k.upper()]}")
            
            if not api_key:
                error_msg = "OPENAI_API_KEY not found. Please set OPENAI_API_KEY in your environment or use fake response mode."
                print(f"❌ {error_msg}")
                return jsonify({
                    "status": "error",
                    "error": error_msg
                }), 400
            else:
                print(f"✅ OPENAI_API_KEY found, proceeding with LLM evaluation...")
        
        try:
            result = evaluate_diversity_with_llm(
                query=query or "Model search query",
                table1=table2_df,  # Model Search (for left side)
                table2=table1_df,  # Table Search (for right side)
                table1_source="Model Search Integration",
                table2_source="Table Search Integration",
                use_fake=use_fake,
                fake_response_path=fake_response_path,
                fake_response_content=fake_response_content
            )
            print(f"✅ Evaluation result success: {result.get('success')}")
        except Exception as eval_error:
            import traceback
            error_traceback = traceback.format_exc()
            print(f"❌ Error in evaluate_diversity_with_llm: {str(eval_error)}")
            print(f"Traceback:\n{error_traceback}")
            return jsonify({
                "status": "error",
                "message": f"Evaluation function error: {str(eval_error)}",
                "traceback": error_traceback
            }), 500
        
        if not result.get("success"):
            error_msg = result.get("error", "Evaluation failed")
            print(f"❌ Evaluation failed: {error_msg}")
            return jsonify({
                "status": "error",
                "message": error_msg,
                "evaluation_result": result
            }), 500
        
        # Save evaluation results
        evaluation_output_path = os.path.join('data', f'evaluation_{job_id}.json')
        os.makedirs('data', exist_ok=True)
        evaluation_result = {
            "job_id": job_id,
            "integration1_path": integration1_path,
            "integration2_path": integration2_path,
            "query": query,
            "evaluation": result,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(evaluation_output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_result, f, ensure_ascii=False, indent=2)
        
        # Also return the two tables for comparison
        # Note: For display consistency, table1 = Model Search (left), table2 = Table Search (right)
        model_search_data = {
            "columns": list(table2_df.columns),
            "data": table2_df.fillna("").astype(str).values.tolist(),
            "shape": list(table2_df.shape)
        }
        table_search_data = {
            "columns": list(table1_df.columns),
            "data": table1_df.fillna("").astype(str).values.tolist(),
            "shape": list(table1_df.shape)
        }
        
        return jsonify({
            "status": "success",
            "job_id": job_id,
            "evaluation": result,
            "table1": {
                "source": "Model Search Integration",
                "data": model_search_data,
                "stats": {
                    "rows": table2_df.shape[0],
                    "columns": table2_df.shape[1],
                    "column_names": list(table2_df.columns)
                }
            },
            "table2": {
                "source": "Table Search Integration",
                "data": table_search_data,
                "stats": {
                    "rows": table1_df.shape[0],
                    "columns": table1_df.shape[1],
                    "column_names": list(table1_df.columns)
                }
            },
            "output_path": evaluation_output_path
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/api/qa', methods=['POST'])
def qa():
    """Answer questions based on integrated table using LLM"""
    try:
        data = request.json or {}
        job_id = data.get('job_id')
        integration_job_id = data.get('integration_job_id')  # Optional: specific integration job_id
        use_table_search = data.get('use_table_search', True)  # Use table search integration by default
        use_fake = data.get('use_fake', False)  # Use fake response for testing
        fake_response_path = data.get('fake_response_path')  # Optional path to fake response file
        fake_response_content = data.get('fake_response_content')  # Optional fake response content
        model = data.get('model', 'gpt-4')  # LLM model to use
        
        if not job_id:
            return jsonify({"status": "error", "message": "job_id is required"}), 400
        
        # Find integration results
        integration_path = None
        data_dir = 'data'
        
        if os.path.exists(data_dir):
            if integration_job_id:
                # Use specific integration job_id
                if use_table_search:
                    # Try to find table search integration
                    for search_type in ['single_column', 'multi_column', 'table_search']:
                        potential_path = os.path.join(data_dir, f'integration_{integration_job_id}_{search_type}.json')
                        if os.path.exists(potential_path):
                            integration_path = potential_path
                            break
                else:
                    # Use model search integration
                    integration_path = os.path.join(data_dir, f'integration_model_search_{integration_job_id}.json')
            else:
                # Auto-discover integration files
                if use_table_search:
                    # Find table search integration
                    for search_type in ['single_column', 'multi_column', 'table_search']:
                        potential_path = os.path.join(data_dir, f'integration_{job_id}_{search_type}.json')
                        if os.path.exists(potential_path):
                            integration_path = potential_path
                            print(f"✅ Found table search integration: {integration_path}")
                            break
                    
                    # If not found, scan all files
                    if not integration_path:
                        for filename in os.listdir(data_dir):
                            if filename.startswith(f'integration_{job_id}_') and filename.endswith('.json'):
                                if not filename.startswith(f'integration_model_search_{job_id}'):
                                    potential_path = os.path.join(data_dir, filename)
                                    if os.path.exists(potential_path):
                                        integration_path = potential_path
                                        print(f"✅ Found table search integration (scanned): {integration_path}")
                                        break
                else:
                    # Use model search integration
                    integration_path = os.path.join(data_dir, f'integration_model_search_{job_id}.json')
        
        # Load integration results
        table_df = None
        query = None
        table_source = "Integrated Table"
        
        if integration_path and os.path.exists(integration_path):
            try:
                with open(integration_path, 'r', encoding='utf-8') as f:
                    integration_data = json.load(f)
                if integration_data.get('integrated_table'):
                    import pandas as pd
                    table_data = integration_data['integrated_table']
                    if 'data' in table_data and 'columns' in table_data:
                        table_df = pd.DataFrame(table_data['data'], columns=table_data['columns'])
                        integration_type_used = integration_data.get('integration_type', 'unknown')
                        search_type_used = integration_data.get('search_type', 'unknown')
                        table_source = f"{'Table Search' if use_table_search else 'Model Search'} Integration ({integration_type_used})"
                        print(f"✅ Loaded table for QA: {table_df.shape[0]} rows × {table_df.shape[1]} columns")
                        print(f"   Integration type: {integration_type_used}, Search type: {search_type_used}")
                    else:
                        print(f"⚠️  Table data missing 'data' or 'columns' keys")
            except Exception as e:
                print(f"❌ Error loading table from {integration_path}: {e}")
                import traceback
                print(traceback.format_exc())
        
        # Get query and model IDs from search results
        # Extract model IDs from BOTH Card2Card and Card2Tab2Card results
        card2card_model_ids = []  # Model IDs from Card2Card search results
        card2tab2card_model_ids = []  # Model IDs from Card2Tab2Card search results
        search_results_path = None
        search_data = None
        
        if not query or not card2card_model_ids or not card2tab2card_model_ids:
            for folder_name in os.listdir(data_dir):
                folder_path = os.path.join(data_dir, folder_name)
                if os.path.isdir(folder_path):
                    potential_path = os.path.join(folder_path, 'search_results.json')
                    if os.path.exists(potential_path):
                        try:
                            with open(potential_path, 'r', encoding='utf-8') as f:
                                search_data = json.load(f)
                            if search_data.get('job_id') == job_id:
                                if not query:
                                    query = search_data.get('query', 'Model search query')
                                
                                # Extract model IDs from Card2Card results
                                card2card_results = search_data.get('card2card_results', [])
                                if card2card_results:
                                    for item in card2card_results:
                                        if isinstance(item, str):
                                            card2card_model_ids.append(item)
                                        elif isinstance(item, dict):
                                            if 'model_id' in item:
                                                card2card_model_ids.append(item['model_id'])
                                            elif 'modelId' in item:
                                                card2card_model_ids.append(item['modelId'])
                                
                                # Extract model IDs from Card2Tab2Card results
                                card2tab2card_results = search_data.get('card2tab2card_results', {})
                                if isinstance(card2tab2card_results, dict):
                                    # Card2Tab2Card results are organized by search type
                                    for search_type, results in card2tab2card_results.items():
                                        if isinstance(results, dict) and 'model_ids' in results:
                                            for model_id in results['model_ids']:
                                                if model_id not in card2tab2card_model_ids:
                                                    card2tab2card_model_ids.append(model_id)
                                        elif isinstance(results, list):
                                            for item in results:
                                                if isinstance(item, str):
                                                    if item not in card2tab2card_model_ids:
                                                        card2tab2card_model_ids.append(item)
                                                elif isinstance(item, dict):
                                                    model_id = item.get('model_id') or item.get('modelId')
                                                    if model_id and model_id not in card2tab2card_model_ids:
                                                        card2tab2card_model_ids.append(model_id)
                                
                                print(f"✅ Found {len(card2card_model_ids)} model IDs from Card2Card results")
                                print(f"✅ Found {len(card2tab2card_model_ids)} model IDs from Card2Tab2Card results")
                                search_results_path = potential_path
                                break
                        except Exception as e:
                            print(f"⚠️  Error reading search results: {e}")
                            import traceback
                            print(traceback.format_exc())
                            continue
        
        # Determine QA mode based on integration type
        qa_mode = "card2tab2card" if use_table_search else "card2card"
        model_ids_to_rank = card2tab2card_model_ids if use_table_search else card2card_model_ids
        
        # For Card2Card mode, table might be optional (can work with just model cards)
        # For Card2Tab2Card mode, table is required
        if qa_mode == "card2tab2card" and (table_df is None or table_df.empty):
            error_msg = f"Could not find integration results for job_id: {job_id}. "
            error_msg += f"Please run Table Integration first."
            if integration_path:
                error_msg += f" Expected file: {integration_path}"
            print(f"❌ {error_msg}")
            return jsonify({
                "status": "error",
                "message": error_msg,
                "integration_path": integration_path,
                "integration_exists": os.path.exists(integration_path) if integration_path else False
            }), 404
        
        # For Card2Card mode, if no table, use empty DataFrame
        if qa_mode == "card2card" and (table_df is None or table_df.empty):
            import pandas as pd
            table_df = pd.DataFrame()  # Empty table, will rely on model card information
            print(f"⚠️  No integrated table found for Card2Card mode, will use model card information only")
        
        if not query:
            query = "Please analyze this integrated table and provide insights."
        
        # Run QA
        print(f"📝 Running QA with use_fake={use_fake}")
        print(f"   Query: {query}")
        print(f"   Table shape: {table_df.shape}")
        
        # Check API key if not using fake
        if not use_fake:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return jsonify({
                    "status": "error",
                    "message": "OPENAI_API_KEY not found. Please set OPENAI_API_KEY in your environment or use fake response mode."
                }), 400
            else:
                print(f"✅ OPENAI_API_KEY found, proceeding with LLM QA...")
        
        try:
            result = answer_question_with_llm(
                query=query,
                table=table_df,
                table_source=table_source,
                qa_mode=qa_mode,  # "card2card" or "card2tab2card"
                model_ids_to_rank=model_ids_to_rank,  # Pass model IDs from search results
                search_results_data=search_data,  # Pass full search results for model card access
                use_fake=use_fake,
                fake_response_path=fake_response_path,
                fake_response_content=fake_response_content,
                model=model
            )
            print(f"   QA Mode: {qa_mode}")
            print(f"   Model IDs to rank: {len(model_ids_to_rank) if model_ids_to_rank else 0}")
            print(f"✅ QA result success: {result.get('success')}")
        except Exception as qa_error:
            import traceback
            error_traceback = traceback.format_exc()
            print(f"❌ Error in answer_question_with_llm: {str(qa_error)}")
            print(f"Traceback:\n{error_traceback}")
            return jsonify({
                "status": "error",
                "message": f"QA function error: {str(qa_error)}",
                "traceback": error_traceback
            }), 500
        
        if not result.get("success"):
            error_msg = result.get("error", "QA failed")
            print(f"❌ QA failed: {error_msg}")
            return jsonify({
                "status": "error",
                "message": error_msg,
                "qa_result": result
            }), 500
        
        # Save QA results
        qa_output_path = os.path.join('data', f'qa_{job_id}_{"table_search" if use_table_search else "model_search"}.json')
        os.makedirs('data', exist_ok=True)
        qa_result = {
            "job_id": job_id,
            "query": query,
            "integration_path": integration_path,
            "table_source": table_source,
            "qa": result,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(qa_output_path, 'w', encoding='utf-8') as f:
            json.dump(qa_result, f, ensure_ascii=False, indent=2)
        
        # Also return the table for reference
        table_data = {
            "columns": list(table_df.columns),
            "data": table_df.fillna("").astype(str).values.tolist(),
            "shape": list(table_df.shape)
        }
        
        return jsonify({
            "status": "success",
            "job_id": job_id,
            "query": query,
            "qa": result,
            "table": {
                "source": table_source,
                "data": table_data,
                "stats": {
                    "rows": table_df.shape[0],
                    "columns": table_df.shape[1],
                    "column_names": list(table_df.columns)
                }
            },
            "output_path": qa_output_path
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
    print("  POST /api/integrate - Integrate tables from Card2Tab2Card (table search) results")
    print("  POST /api/integrate-model-search - Integrate tables from Card2Card (model search) results")
    print("  POST /api/evaluate - Evaluate diversity between two integrated tables using LLM")
    print("  POST /api/qa - Answer questions based on integrated table using LLM")
    print("\nAll results saved to data/ directory")
    app.run(host='0.0.0.0', port=5002, debug=False, threaded=True)
