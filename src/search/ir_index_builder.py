"""
Build dense (embedding .npz) and sparse (Pyserini Lucene) indexes for ModelCard retrieval.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from typing import List

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import duckdb
import numpy as np
from tqdm import tqdm

from src.config import (
    CARD2CARD_SPARSE_CORPUS,
    EMB_NPZ,
    ENCODE_MODEL,
    MODELCARD_STEP1_PARQUET,
    SPARSE_INDEX,
)
from src.utils import get_device

_SQL_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _quote_sql_ident(name: str) -> str:
    if not _SQL_IDENT.match(name):
        raise ValueError(f"Invalid SQL column name: {name!r} (use letters, digits, underscore)")
    return f'"{name}"'


def _ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


class DenseCardIndexBuilder:
    """
    Stream ``modelId`` + ``card_readme`` from parquet, encode with SentenceTransformer, write ``.npz``.
    """

    def __init__(
        self,
        *,
        parquet_path: str | None = None,
        output_npz: str | None = None,
        encode_model: str | None = None,
    ) -> None:
        self.parquet_path = parquet_path or MODELCARD_STEP1_PARQUET
        self.output_npz = output_npz or EMB_NPZ
        self.encode_model = encode_model or ENCODE_MODEL

    def build(self, batch_size: int = 256) -> None:
        if not os.path.isfile(self.parquet_path):
            raise FileNotFoundError(f"Parquet not found: {self.parquet_path}")

        _ensure_parent_dir(self.output_npz)
        id_q = _quote_sql_ident("modelId")
        text_q = _quote_sql_ident("card_readme")
        sql = f"""
            SELECT CAST({id_q} AS VARCHAR) AS id,
                   CAST({text_q} AS VARCHAR) AS txt
            FROM read_parquet(?)
            WHERE {text_q} IS NOT NULL
              AND length(trim(cast({text_q} AS VARCHAR))) > 0
        """

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.encode_model, device=get_device())
        model.eval()

        all_embs: List[np.ndarray] = []
        ids: List[str] = []
        batch_ids: List[str] = []
        batch_texts: List[str] = []

        con = duckdb.connect(":memory:")
        try:
            result = con.execute(sql, [[self.parquet_path]])
            reader = result.fetch_record_batch(4096)
            pbar = tqdm(unit="rows", desc="Encoding (SQL stream)")
            for record_batch in reader:
                col_id = record_batch.column(0)
                col_txt = record_batch.column(1)
                row_ids = col_id.to_pylist()
                row_txts = col_txt.to_pylist()
                for mid, txt in zip(row_ids, row_txts):
                    if mid is None or txt is None:
                        continue
                    s_id = str(mid).strip()
                    s_txt = str(txt).strip()
                    if not s_id or not s_txt:
                        continue
                    batch_ids.append(s_id)
                    batch_texts.append(s_txt)
                    pbar.update(1)
                    if len(batch_texts) >= batch_size:
                        try:
                            embs = model.encode(
                                batch_texts,
                                convert_to_numpy=True,
                                show_progress_bar=False,
                                batch_size=len(batch_texts),
                            )
                            if embs is not None and getattr(embs, "size", 0) > 0:
                                all_embs.append(np.asarray(embs, dtype=np.float32))
                                ids.extend(batch_ids)
                        except Exception as e:
                            print(f"Error encoding batch (last id={batch_ids[-1]!r}): {e}")
                        finally:
                            batch_ids = []
                            batch_texts = []
            pbar.close()

            if batch_texts:
                try:
                    embs = model.encode(
                        batch_texts,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        batch_size=len(batch_texts),
                    )
                    if embs is not None and getattr(embs, "size", 0) > 0:
                        all_embs.append(np.asarray(embs, dtype=np.float32))
                        ids.extend(batch_ids)
                except Exception as e:
                    print(f"Error encoding final batch: {e}")
        finally:
            con.close()

        if not all_embs:
            print("No embeddings generated, skipping save.")
            return
        embs_array = np.vstack(all_embs).astype(np.float32, copy=False)
        np.savez_compressed(self.output_npz, embeddings=embs_array, ids=np.array(ids, dtype=str))
        print(f"Saved embeddings: {self.output_npz}, shape={embs_array.shape}, n_ids={len(ids)}")


class SparseCardIndexBuilder:
    """
    Write corpus JSONL from parquet, then run Pyserini Lucene indexer (BM25).
    """

    def __init__(
        self,
        *,
        parquet_path: str | None = None,
        corpus_parent_dir: str | None = None,
        output_lucene_dir: str | None = None,
    ) -> None:
        self.parquet_path = parquet_path or MODELCARD_STEP1_PARQUET
        self.corpus_parent_dir = corpus_parent_dir or CARD2CARD_SPARSE_CORPUS
        self.output_lucene_dir = output_lucene_dir or SPARSE_INDEX

    def build(self, threads: int = 1) -> None:
        if not os.path.isfile(self.parquet_path):
            raise FileNotFoundError(f"Parquet not found: {self.parquet_path}")

        os.makedirs(self.corpus_parent_dir, exist_ok=True)
        corpus_jsonl = os.path.join(self.corpus_parent_dir, "corpus.jsonl")
        print(f"Building corpus JSONL from parquet via SQL: {corpus_jsonl}")

        id_q = _quote_sql_ident("modelId")
        text_q = _quote_sql_ident("card_readme")
        sql = f"""
            SELECT CAST({id_q} AS VARCHAR) AS id,
                   CAST({text_q} AS VARCHAR) AS txt
            FROM read_parquet(?)
            WHERE {text_q} IS NOT NULL
              AND length(trim(cast({text_q} AS VARCHAR))) > 0
        """
        with duckdb.connect(":memory:") as con:
            result = con.execute(sql, [[self.parquet_path]])
            reader = result.fetch_record_batch(8192)
            written = 0
            with open(corpus_jsonl, "w", encoding="utf-8") as f:
                for record_batch in tqdm(reader, desc="Write corpus.jsonl (SQL stream)", unit="batch"):
                    row_ids = record_batch.column(0).to_pylist()
                    row_txts = record_batch.column(1).to_pylist()
                    for mid, txt in zip(row_ids, row_txts):
                        if mid is None or txt is None:
                            continue
                        s_id = str(mid).strip()
                        s_txt = str(txt).strip()
                        if not s_id or not s_txt:
                            continue
                        f.write(json.dumps({"id": s_id, "contents": s_txt}, ensure_ascii=False) + "\n")
                        written += 1
        if written == 0:
            raise RuntimeError("No rows written to corpus.jsonl (check parquet path / column names / filters).")
        print(f"✅ Wrote {written} docs: {corpus_jsonl}")
        print("Building Lucene index (BM25, same as ModelTables baseline2)...")
        t0 = time.time()
        _ensure_parent_dir(self.output_lucene_dir)
        cmd = [
            sys.executable,
            "-m",
            "pyserini.index.lucene",
            "--collection",
            "JsonCollection",
            "--input",
            os.path.abspath(self.corpus_parent_dir),
            "--index",
            os.path.abspath(self.output_lucene_dir),
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
        print(f"✅ Sparse index saved: {self.output_lucene_dir} (Total: {time.time() - t0:.2f}s)")

def main() -> None:
    parser = argparse.ArgumentParser(description="Build ModelCard dense (.npz) + sparse (Lucene) indexes")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command")

    build_parser = subparsers.add_parser("build-dense-index", help="Encode card_readme → embeddings .npz")
    build_parser.add_argument("--batch_size", type=int, default=256)

    sparse_build_parser = subparsers.add_parser(
        "build-sparse-index", help="Parquet → corpus.jsonl → Pyserini Lucene index"
    )
    sparse_build_parser.add_argument("--threads", type=int, default=1)

    args = parser.parse_args()
    t0 = time.time()

    if args.command == "build-dense-index":
        print(
            f"[ir_index_builder] build-dense-index | encode_model={ENCODE_MODEL!r} | "
            f"device={get_device()!r} | batch_size={args.batch_size}",
            flush=True,
        )
        DenseCardIndexBuilder().build(batch_size=args.batch_size)
    elif args.command == "build-sparse-index":
        print(f"[ir_index_builder] build-sparse-index | threads={args.threads}", flush=True)
        SparseCardIndexBuilder().build(threads=args.threads)

    print(f"\nTotal time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()