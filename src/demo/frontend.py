"""
Frontend CLI for ModelSearch Demo

Interactive command-line interface for testing search functionality.
"""

import os
import sys
import json
import argparse
from typing import Optional

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.search import (
    build_card_index,
    search_card2card,
    search_query2modelcard,
    search_table2table,
    search_card2tab2card
)


def interactive_menu():
    """Interactive menu for testing search functions"""
    print("=" * 60)
    print("ModelSearch Interactive Demo")
    print("=" * 60)
    print("\nAvailable commands:")
    print("  1. Build Index (build embeddings)")
    print("  2. Query to ModelCard Search")
    print("  3. Card to Card Search")
    print("  4. Table to Table Search (testing)")
    print("  5. Card to Tab to Card Search")
    print("  6. Exit")
    print("\nAll results saved to data/ directory")
    print("=" * 60)
    
    while True:
        try:
            choice = input("\nEnter command number (1-6): ").strip()
            
            if choice == '1':
                build_index_interactive()
            elif choice == '2':
                query2modelcard_interactive()
            elif choice == '3':
                card2card_interactive()
            elif choice == '4':
                tab2tab_interactive()
            elif choice == '5':
                card2tab2card_interactive()
            elif choice == '6':
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please enter 1-6.")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


def build_index_interactive():
    """Interactive build index"""
    print("\n--- Build Index ---")
    field = input("Field to use (card/card_readme) [card]: ").strip() or "card"
    raw_dir = input("Raw data directory [data_citationlake/raw]: ").strip() or "data_citationlake/raw"
    device = input("Device (cuda/cpu) [cuda]: ").strip() or "cuda"
    
    print("\nBuilding index...")
    build_card_index(
        field=field,
        raw_dir=raw_dir,
        output_jsonl="data/card2card_corpus.jsonl",
        output_npz="data/card2card_embeddings.npz",
        output_index="data/card2card.faiss",
        device=device
    )
    print("✅ Index built successfully!")


def query2modelcard_interactive():
    """Interactive query to modelcard search"""
    print("\n--- Query to ModelCard Search ---")
    query = input("Enter search query: ").strip()
    if not query:
        print("Query cannot be empty")
        return
    
    top_k = input("Number of results [20]: ").strip()
    top_k = int(top_k) if top_k else 20
    
    device = input("Device (cuda/cpu) [cuda]: ").strip() or "cuda"
    output_json = "data/query2modelcard_results.json"
    
    print(f"\nSearching for: '{query}'...")
    results = search_query2modelcard(
        query=query,
        emb_npz="data/card2card_embeddings.npz",
        faiss_index="data/card2card.faiss",
        top_k=top_k,
        device=device,
        output_json=output_json
    )
    
    print(f"\n✅ Found {len(results)} results:")
    for i, model_id in enumerate(results, 1):
        print(f"  {i}. {model_id}")
    print(f"\nResults saved to {output_json}")


def card2card_interactive():
    """Interactive card to card search"""
    print("\n--- Card to Card Search ---")
    model_id = input("Enter model ID (e.g., Salesforce/codet5-base): ").strip()
    if not model_id:
        print("Model ID cannot be empty")
        return
    
    top_k = input("Number of results [20]: ").strip()
    top_k = int(top_k) if top_k else 20
    output_json = "data/card2card_results.json"
    
    print(f"\nSearching for similar models to '{model_id}'...")
    results = search_card2card(
        model_id=model_id,
        emb_npz="data/card2card_embeddings.npz",
        faiss_index="data/card2card.faiss",
        top_k=top_k,
        output_json=output_json
    )
    
    print(f"\n✅ Found {len(results)} similar models:")
    for i, model_id_result in enumerate(results, 1):
        print(f"  {i}. {model_id_result}")
    print(f"\nResults saved to {output_json}")


def tab2tab_interactive():
    """Interactive table to table search (testing)"""
    print("\n--- Table to Table Search (Testing) ---")
    print("Note: This requires modellake.db to be set up")
    
    search_type = input("Search type (single_column/multi_column/keyword) [keyword]: ").strip() or "keyword"
    
    if search_type == 'multi_column':
        query_path = input("Enter path to CSV file: ").strip()
        if not os.path.exists(query_path):
            print(f"File not found: {query_path}")
            return
        import pandas as pd
        query = pd.read_csv(query_path)
    else:
        query_str = input("Enter query (comma-separated for values/keywords): ").strip()
        if not query_str:
            print("Query cannot be empty")
            return
        query = [x.strip() for x in query_str.split(',')]
    
    k = input("Number of results [10]: ").strip()
    k = int(k) if k else 10
    output_json = "data/tab2tab_results.json"
    
    print(f"\nSearching tables...")
    results = search_table2table(query, search_type, k)
    
    print(f"\n✅ Found {len(results)} tables:")
    for i, table_id in enumerate(results, 1):
        print(f"  {i}. Table ID: {table_id}")
    
    # Save results
    result_data = {
        "query": query if isinstance(query, list) else str(query),
        "search_type": search_type,
        "k": k,
        "results": [int(tid) for tid in results],
        "num_results": len(results)
    }
    os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else '.', exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {output_json}")


def card2tab2card_interactive():
    """Interactive card to tab to card search"""
    print("\n--- Card to Tab to Card Search ---")
    model_id = input("Enter model ID: ").strip()
    if not model_id:
        print("Model ID cannot be empty")
        return
    
    query = input("Enter query (comma-separated keywords, or press Enter to use model's tables): ").strip()
    search_type = input("Search type (single_column/multi_column/keyword) [keyword]: ").strip() or "keyword"
    k = input("Number of results [10]: ").strip()
    k = int(k) if k else 10
    
    schema_log = input("Schema log path [data_citationlake/logs/parquet_schema.log]: ").strip() or "data_citationlake/logs/parquet_schema.log"
    output_json = "data/card2tab2card_results.json"
    
    # Parse query if provided
    query_parsed = None
    if query:
        if search_type == 'keyword':
            query_parsed = [x.strip() for x in query.split(',')]
        elif search_type == 'single_column':
            query_parsed = [x.strip() for x in query.split(',')]
        elif search_type == 'multi_column':
            if not os.path.exists(query):
                print(f"File not found: {query}")
                return
            import pandas as pd
            query_parsed = pd.read_csv(query)
    
    print(f"\nSearching for models similar to '{model_id}' via table search...")
    results = search_card2tab2card(
        model_id=model_id,
        query=query_parsed,
        search_type=search_type,
        k=k,
        schema_log_path=schema_log,
        use_citationlake=True,
        output_json=output_json
    )
    
    print(f"\n✅ Found {len(results)} similar models:")
    for i, model_id_result in enumerate(results, 1):
        print(f"  {i}. {model_id_result}")
    print(f"\nResults saved to {output_json}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="ModelSearch Interactive Demo")
    parser.add_argument('--non-interactive', action='store_true',
                       help='Run in non-interactive mode (for scripting)')
    
    args = parser.parse_args()
    
    if args.non_interactive:
        print("Non-interactive mode. Use individual search functions directly.")
        print("Example:")
        print("  from src.search import search_query2modelcard")
        print("  results = search_query2modelcard(query='transformer model')")
    else:
        interactive_menu()


if __name__ == '__main__':
    main()

