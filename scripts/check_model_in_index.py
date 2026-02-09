#!/usr/bin/env python3
"""
Check if a model_id exists in the card2card embeddings index (ids in npz).
Exit 0 if found, 1 if not. Used by backend to reject user-provided IDs not in dataset.
"""
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Check if model_id is in card2card index")
    parser.add_argument("--model_id", required=True, help="HuggingFace model ID")
    parser.add_argument("--emb_npz", default="data/card2card_embeddings.npz", help="Path to embeddings npz (has 'ids')")
    args = parser.parse_args()
    try:
        import numpy as np
    except ImportError:
        print("numpy required", file=sys.stderr)
        sys.exit(2)
    if not __import__("os").path.exists(args.emb_npz):
        print(f"emb_npz not found: {args.emb_npz}", file=sys.stderr)
        sys.exit(2)
    data = np.load(args.emb_npz, allow_pickle=True)
    ids = data.get("ids")
    if ids is None:
        print("npz has no 'ids' key", file=sys.stderr)
        sys.exit(2)
    ids_list = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    if args.model_id in ids_list:
        print("ok")
        sys.exit(0)
    print(f"Model ID '{args.model_id}' not in dataset (not in {args.emb_npz})", file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    main()
