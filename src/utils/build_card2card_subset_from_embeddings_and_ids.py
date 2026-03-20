"""
Build dense + sparse (Lucene BM25) subset artifacts for card2card search.

Inputs:
  - an existing embeddings npz (dense vectors for all model cards)
  - a txt of allowed model ids (one id per line)

Outputs:
  - a filtered embeddings npz (dense retrieval subset)
  - a filtered Pyserini Lucene index (sparse retrieval subset)

This lets you run `src.search.card2card search` on a smaller candidate space
by passing:
  - --embeddings_npz_path=<subset_dense_npz>
  - --sparse_index_path=<subset_sparse_index_dir>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Set, Tuple

import duckdb
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import (
    EMB_NPZ,
    EMB_NPZ_HUGGING,
    SPARSE_INDEX_HUGGING,
    CARD2CARD_SPARSE_CORPUS,
    CARD2CARD_SPARSE_CORPUS_HUGGING,
    MODELCARD_STEP1_PARQUET,
)


def _load_txt_ids(path: str) -> List[str]:
    ids: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                ids.append(s)
    return ids


def _load_embeddings_npz(path: str) -> Tuple[np.ndarray, List[str]]:
    data = np.load(path, allow_pickle=True)
    embs = np.asarray(data["embeddings"], dtype=np.float32)
    ids = data["ids"].tolist()
    return embs, ids


def build_dense_subset(
    *,
    embeddings_npz_path: str,
    model_ids_txt: str,
    output_dense_npz_path: str,
) -> List[str]:
    embs, all_ids = _load_embeddings_npz(embeddings_npz_path)
    id_to_idx = {mid: i for i, mid in enumerate(all_ids)}

    requested_ids = _load_txt_ids(model_ids_txt)
    requested_set: Set[str] = set(requested_ids)

    present_ids_in_order: List[str] = [mid for mid in requested_ids if mid in id_to_idx]
    missing = requested_set - set(present_ids_in_order)

    if not present_ids_in_order:
        raise RuntimeError(
            "Dense subset: none of the ids from model_ids_txt are present in embeddings npz. "
            f"Requested={len(requested_set)}, Present={len(present_ids_in_order)}, Missing~={len(missing)}"
        )

    indices = [id_to_idx[mid] for mid in present_ids_in_order]
    subset_embs = embs[indices]

    os.makedirs(os.path.dirname(output_dense_npz_path) or ".", exist_ok=True)
    np.savez_compressed(output_dense_npz_path, embeddings=np.asarray(subset_embs, dtype=np.float32), ids=np.array(present_ids_in_order, dtype=str))

    print(f"[dense] wrote subset embeddings: {output_dense_npz_path}\nrequested_ids={len(requested_set)} present_in_embeddings={len(present_ids_in_order)} missing~={len(missing)}")
    return present_ids_in_order


def build_sparse_subset(
    *,
    subset_ids: List[str],
    output_sparse_index_dir: str,
    full_corpus_jsonl_path: str,
    threads: int,
    corpus_dir: str,
) -> None:
    # Pyserini JsonCollection expects an input directory containing corpus.jsonl.
    output_dir = os.path.dirname(os.path.abspath(output_sparse_index_dir))
    os.makedirs(output_dir, exist_ok=True)

    corpus_jsonl_path = os.path.join(corpus_dir, "corpus.jsonl")
    print(f"[sparse] writing subset corpus.jsonl: {corpus_jsonl_path}")
    subset_id_set = set(subset_ids)

    wrote = 0
    # Fast path: filter from existing full sparse corpus jsonl.
    print(f"[sparse] filtering from full corpus jsonl: {full_corpus_jsonl_path}")
    with open(full_corpus_jsonl_path, "r", encoding="utf-8") as src, open(
        corpus_jsonl_path, "w", encoding="utf-8"
    ) as dst:
        for line in tqdm(src, desc="[sparse] filter corpus.jsonl", unit="line"):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            mid = str(obj.get("id", "")).strip()
            if mid and mid in subset_id_set:
                dst.write(json.dumps({"id": mid, "contents": obj.get("contents", "")}, ensure_ascii=False) + "\n")
                wrote += 1
    
    if wrote == 0:
        raise RuntimeError("Sparse subset corpus is empty (0 docs). Check model_ids_txt / corpus source.")

    # Build Lucene index using Pyserini.
    print(f"[sparse] building Lucene index: {output_sparse_index_dir}")
    cmd = [
        sys.executable,
        "-m",
        "pyserini.index.lucene",
        "--collection",
        "JsonCollection",
        "--input",
        os.path.abspath(corpus_dir),
        "--index",
        os.path.abspath(output_sparse_index_dir),
        "--generator",
        "DefaultLuceneDocumentGenerator",
        "--threads",
        str(threads),
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"pyserini.index.lucene failed: {out.stderr or out.stdout}")
    print("[sparse] ✅ subset Lucene index built.")

    try:
        shutil.rmtree(corpus_dir, ignore_errors=True)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hugging subset artifacts for card2card search.")
    parser.add_argument("--model_ids_txt", required=True, help="Txt file: one model id per line (typically generated by build_valid_model_ids_txt --resources hugging).")
    parser.add_argument("--threads", type=int, default=1, help="Pyserini indexing threads.")
    args = parser.parse_args()

    ids_txt_path = os.path.abspath(args.model_ids_txt)
    output_dense_npz_path = EMB_NPZ_HUGGING
    output_sparse_index_dir = SPARSE_INDEX_HUGGING
    corpus_dir = CARD2CARD_SPARSE_CORPUS_HUGGING
    full_corpus_jsonl_path = os.path.join(CARD2CARD_SPARSE_CORPUS, "corpus.jsonl")
    emb_npz_path = EMB_NPZ

    subset_ids = build_dense_subset(embeddings_npz_path=emb_npz_path, model_ids_txt=ids_txt_path, output_dense_npz_path=output_dense_npz_path)
    build_sparse_subset(subset_ids=subset_ids, output_sparse_index_dir=output_sparse_index_dir, full_corpus_jsonl_path=full_corpus_jsonl_path, threads=args.threads, corpus_dir=corpus_dir)

if __name__ == "__main__":
    main()

