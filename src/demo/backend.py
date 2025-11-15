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
    search_card2tab2card
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
        """Add log message"""
        with self.lock:
            timestamp = datetime.now().isoformat()
            self.logs.append({"timestamp": timestamp, "message": message})
            print(f"[{self.job_id}] {message}")
    
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


def run_search_pipeline(job_id: str, query: str, top_k: int = 20):
    """Run the complete search pipeline in background"""
    logger = jobs[job_id]
    
    try:
        logger.set_status("running")
        logger.log("Starting search pipeline...")
        logger.log(f"Query: {query}")
        
        # Step 1: Extract model card from query
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
        
        # Step 2: Prepare query CSV for Card2Tab2Card
        logger.log("Step 2: Preparing query CSV for table search...")
        query_csv = None
        
        # Try to find a CSV file from the model's tables
        default_csvs = [
            "data_citationlake/processed/deduped_hugging_csvs/0000e35dae_table1.csv",
            "data_citationlake/processed/deduped_github_csvs/0021c79d4e1a37579ca87328864d67a5_table_0.csv"
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
                results = search_card2card(
                    model_id=model_id,
                    emb_npz=DEFAULT_EMB_NPZ,
                    faiss_index=DEFAULT_FAISS_INDEX,
                    top_k=top_k,
                    output_json=None
                )
                logger.log(f"  ✅ [Card2Card] Found {len(results)} results")
                return results
            except Exception as e:
                logger.log(f"  ❌ [Card2Card] Error: {str(e)}")
                return {"error": str(e)}
        
        def run_card2tab2card_search(search_type_name, query_parsed):
            """Run one Card2Tab2Card search type"""
            logger.log(f"  [Card2Tab2Card-{search_type_name}] Starting...")
            try:
                results = search_card2tab2card(
                    model_id=model_id,
                    relationship_parquet=DEFAULT_RELATIONSHIP_PARQUET,
                    query=query_parsed,
                    search_type=search_type_name,
                    k=top_k,
                    schema_log_path=DEFAULT_SCHEMA_LOG,
                    use_citationlake=True,
                    output_json=None,
                    db_path=DEFAULT_DB_PATH
                )
                logger.log(f"  ✅ [Card2Tab2Card-{search_type_name}] Found {len(results)} results")
                return search_type_name, results
            except Exception as e:
                logger.log(f"  ❌ [Card2Tab2Card-{search_type_name}] Error: {str(e)}")
                return search_type_name, {"error": str(e)}
        
        # Prepare queries for all three search types
        queries_parsed = {}
        if query_df is not None:
            first_col = query_df.columns[0]
            queries_parsed['single_column'] = query_df[first_col].dropna().astype(str).tolist()
            queries_parsed['keyword'] = query_df[first_col].dropna().astype(str).tolist()
            queries_parsed['unionable'] = query_df
        else:
            queries_parsed['single_column'] = None
            queries_parsed['keyword'] = None
            queries_parsed['unionable'] = None
        
        # Run all searches in parallel using ThreadPoolExecutor
        card2card_results = None
        card2tab2card_all = {}
        
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
                    query_parsed
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
            if isinstance(tab2card_results, list):
                tab2card_set = set(tab2card_results)
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
        
        # Save results
        output_json = os.path.join('data', f'compare_search_{job_id}.json')
        os.makedirs('data', exist_ok=True)
        results_data = {
            "job_id": job_id,
            "query": query,
            "model_id": model_id,
            "top_k": top_k,
            "card2card_results": card2card_results,
            "card2tab2card_results": card2tab2card_all,
            "comparison": comparison,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        
        logger.log(f"✅ Results saved to {output_json}")
        logger.set_results(results_data)
        logger.log("Pipeline completed successfully!")
        
    except Exception as e:
        logger.log(f"❌ Error: {str(e)}")
        logger.set_status("error")
        import traceback
        logger.log(traceback.format_exc())


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok"})


@app.route('/api/search', methods=['POST'])
def search():
    """Main search endpoint - starts pipeline"""
    try:
        data = request.json or {}
        query = data.get('query')
        if not query:
            return jsonify({"status": "error", "message": "query is required"}), 400
        
        top_k = data.get('top_k', 20)
        
        # Create job
        job_id = str(uuid.uuid4())
        jobs[job_id] = JobLogger(job_id)
        
        # Start pipeline in background thread
        thread = threading.Thread(
            target=run_search_pipeline,
            args=(job_id, query, top_k)
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
