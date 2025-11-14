"""
Backend API for ModelSearch Demo

Provides REST API endpoints for search functionality.
All results are saved to data/ directory.
"""

import os
import sys
import json
from typing import Dict, List, Optional, Any
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.search import (
    build_card_index,
    search_card2card,
    search_query2modelcard,
    search_table2table,
    search_card2tab2card
)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Default paths (all in data/)
DEFAULT_EMB_NPZ = "data/card2card_embeddings.npz"
DEFAULT_FAISS_INDEX = "data/card2card.faiss"
DEFAULT_SCHEMA_LOG = "data_citationlake/logs/parquet_schema.log"


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok"})


@app.route('/api/build-index', methods=['POST'])
def build_index():
    """Build FAISS index from model cards"""
    try:
        data = request.json or {}
        field = data.get('field', 'card')
        raw_dir = data.get('raw_dir', 'data_citationlake/raw')
        parquet = data.get('parquet', None)
        device = data.get('device', 'cuda')
        
        build_card_index(
            field=field,
            raw_dir=raw_dir,
            parquet=parquet,
            output_jsonl="data/card2card_corpus.jsonl",
            output_npz=DEFAULT_EMB_NPZ,
            output_index=DEFAULT_FAISS_INDEX,
            device=device
        )
        
        return jsonify({
            "status": "success",
            "message": "Index built successfully",
            "embeddings": DEFAULT_EMB_NPZ,
            "index": DEFAULT_FAISS_INDEX
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/query2modelcard', methods=['POST'])
def query2modelcard():
    """Search model cards using text query"""
    try:
        data = request.json or {}
        query = data.get('query')
        if not query:
            return jsonify({"status": "error", "message": "query is required"}), 400
        
        top_k = data.get('top_k', 20)
        device = data.get('device', 'cuda')
        output_json = data.get('output_json', 'data/query2modelcard_results.json')
        
        results = search_query2modelcard(
            query=query,
            emb_npz=DEFAULT_EMB_NPZ,
            faiss_index=DEFAULT_FAISS_INDEX,
            top_k=top_k,
            device=device,
            output_json=output_json
        )
        
        return jsonify({
            "status": "success",
            "query": query,
            "results": results,
            "count": len(results),
            "output_file": output_json
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/card2card', methods=['POST'])
def card2card():
    """Search for similar model cards"""
    try:
        data = request.json or {}
        model_id = data.get('model_id')
        if not model_id:
            return jsonify({"status": "error", "message": "model_id is required"}), 400
        
        top_k = data.get('top_k', 20)
        output_json = data.get('output_json', 'data/card2card_results.json')
        
        results = search_card2card(
            model_id=model_id,
            emb_npz=DEFAULT_EMB_NPZ,
            faiss_index=DEFAULT_FAISS_INDEX,
            top_k=top_k,
            output_json=output_json
        )
        
        return jsonify({
            "status": "success",
            "query_model": model_id,
            "results": results,
            "count": len(results),
            "output_file": output_json
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/tab2tab', methods=['POST'])
def tab2tab():
    """Table to table search (testing tool)"""
    try:
        data = request.json or {}
        search_type = data.get('search_type', 'keyword')
        query = data.get('query')
        if not query:
            return jsonify({"status": "error", "message": "query is required"}), 400
        
        k = data.get('k', 10)
        output_json = data.get('output_json', 'data/tab2tab_results.json')
        
        # Parse query based on search type
        if search_type == 'single_column':
            query_list = query if isinstance(query, list) else [x.strip() for x in str(query).split(',')]
        elif search_type == 'multi_column':
            # For multi_column, query should be a CSV path
            import pandas as pd
            query_list = pd.read_csv(query)
        elif search_type == 'keyword':
            query_list = query if isinstance(query, list) else [x.strip() for x in str(query).split(',')]
        else:
            return jsonify({"status": "error", "message": f"Unknown search_type: {search_type}"}), 400
        
        results = search_table2table(query_list, search_type, k)
        
        # Save results
        os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else '.', exist_ok=True)
        result_data = {
            "query": query_list if isinstance(query_list, list) else str(query_list),
            "search_type": search_type,
            "k": k,
            "results": [int(tid) for tid in results],
            "num_results": len(results)
        }
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            "status": "success",
            "query": query_list if isinstance(query_list, list) else str(query_list),
            "search_type": search_type,
            "results": [int(tid) for tid in results],
            "count": len(results),
            "output_file": output_json
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/card2tab2card', methods=['POST'])
def card2tab2card():
    """Card to tab to card search"""
    try:
        data = request.json or {}
        model_id = data.get('model_id')
        if not model_id:
            return jsonify({"status": "error", "message": "model_id is required"}), 400
        
        query = data.get('query', None)
        search_type = data.get('search_type', 'keyword')
        k = data.get('k', 10)
        schema_log = data.get('schema_log', DEFAULT_SCHEMA_LOG)
        use_citationlake = data.get('use_citationlake', True)
        output_json = data.get('output_json', 'data/card2tab2card_results.json')
        
        # Parse query if provided
        query_parsed = None
        if query:
            if search_type == 'single_column':
                query_parsed = query if isinstance(query, list) else [x.strip() for x in str(query).split(',')]
            elif search_type == 'multi_column':
                import pandas as pd
                query_parsed = pd.read_csv(query)
            elif search_type == 'keyword':
                query_parsed = query if isinstance(query, list) else [x.strip() for x in str(query).split(',')]
        
        results = search_card2tab2card(
            model_id=model_id,
            relationship_parquet=data.get('relationship_parquet', None),
            query=query_parsed,
            search_type=search_type,
            k=k,
            schema_log_path=schema_log,
            use_citationlake=use_citationlake,
            output_json=output_json
        )
        
        return jsonify({
            "status": "success",
            "query_model": model_id,
            "results": results,
            "count": len(results),
            "output_file": output_json
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    print("Starting ModelSearch Backend API...")
    print("Endpoints:")
    print("  POST /api/build-index - Build FAISS index")
    print("  POST /api/query2modelcard - Search with text query")
    print("  POST /api/card2card - Search similar model cards")
    print("  POST /api/tab2tab - Table to table search (testing)")
    print("  POST /api/card2tab2card - Card to tab to card search")
    print("\nAll results saved to data/ directory")
    app.run(host='0.0.0.0', port=5000, debug=True)

