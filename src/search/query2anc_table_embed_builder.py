"""Build dense table embeddings for query-to-anchor-table search."""

from __future__ import annotations

import os
from typing import List, Sequence

import duckdb
import numpy as np
from tqdm import tqdm

from src.config import ENCODE_MODEL, INDEX_TABLE, MODEL_TO_TABLES_EXPLODE_PARQUET, OUTPUT_DIR
from src.utils import _paths_for_resource_set, get_device


def default_table_embedding_npz(resources: Sequence[str]) -> str:
    rset = {str(r).strip().lower() for r in resources if str(r).strip()}
    if rset == {"hugging"}:
        return os.path.join(OUTPUT_DIR, "query2anc_table_embeddings_hugging.npz")
    if rset == {"hugging", "github", "arxiv"}:
        return os.path.join(OUTPUT_DIR, "query2anc_table_embeddings.npz")
    suffix = "_".join(sorted(rset)) or "custom"
    return os.path.join(OUTPUT_DIR, f"query2anc_table_embeddings_{suffix}.npz")


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _sql_string(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


class Query2AnchorTableEmbeddingBuilder:
    """Encode table tokens into the same .npz format consumed by DenseSearcher."""

    def __init__(
        self,
        *,
        resources: Sequence[str] | None = None,
        output_npz: str | None = None,
        db_path: str | None = None,
        encode_model: str | None = None,
    ) -> None:
        self.resources = [str(r).strip().lower() for r in (resources or ["hugging"]) if str(r).strip()]
        self.output_npz = output_npz or default_table_embedding_npz(self.resources)
        self.db_path = db_path or _paths_for_resource_set(self.resources)[2]
        self.encode_model = encode_model or ENCODE_MODEL

    def _table_text_sql(self) -> str:
        resource_values = ", ".join(_sql_string(r) for r in self.resources)
        resource_sql = f"AND e.resource IN ({resource_values})" if resource_values else ""
        explode_path = os.path.abspath(MODEL_TO_TABLES_EXPLODE_PARQUET).replace("\\", "/")
        return f"""
        WITH allowed AS (
            SELECT DISTINCT CAST(e.csv_basename AS VARCHAR) AS csv_basename
            FROM read_parquet('{explode_path}') AS e
            WHERE e.csv_basename IS NOT NULL
              AND length(trim(CAST(e.csv_basename AS VARCHAR))) > 0
              {resource_sql}
        ),
        indexed AS (
            SELECT
                regexp_extract(CAST(c.filename AS VARCHAR), '[^/]+$') AS csv_basename,
                c.tokenized,
                c.rowid,
                c.colid
            FROM {INDEX_TABLE} AS c
            WHERE c.table_type = 'ori'
              AND c.tokenized IS NOT NULL
              AND length(trim(CAST(c.tokenized AS VARCHAR))) > 0
        )
        SELECT
            i.csv_basename,
            string_agg(CAST(i.tokenized AS VARCHAR), ' ' ORDER BY i.rowid, i.colid) AS text
        FROM indexed AS i
        INNER JOIN allowed AS a
          ON a.csv_basename = i.csv_basename
        GROUP BY i.csv_basename
        ORDER BY i.csv_basename
        """

    def _count_table_texts(self, con_data: duckdb.DuckDBPyConnection) -> int:
        sql = f"SELECT count(*) FROM ({self._table_text_sql()}) AS table_texts"
        return int(con_data.execute(sql).fetchone()[0])

    def _iter_table_texts(self, con_data: duckdb.DuckDBPyConnection, batch_size: int):
        sql = self._table_text_sql()
        result = con_data.execute(sql)
        while True:
            rows = result.fetchmany(batch_size)
            if not rows:
                break
            ids = [str(row[0]).strip() for row in rows if row and str(row[0]).strip()]
            texts = [str(row[1] or "").strip() for row in rows if row and str(row[0]).strip()]
            yield ids, texts

    def build(self, batch_size: int = 256) -> None:
        if not os.path.isfile(self.db_path):
            raise FileNotFoundError(f"ModelLake DuckDB not found: {self.db_path}")

        _ensure_parent_dir(self.output_npz)
        from sentence_transformers import SentenceTransformer

        print(
            f"[query2anc_table_embed_pre] encode_model={self.encode_model!r} | "
            f"device={get_device()!r} | resources={self.resources!r} | batch_size={batch_size}",
            flush=True,
        )
        model = SentenceTransformer(self.encode_model, device=get_device())
        model.eval()

        all_embs: List[np.ndarray] = []
        ids: List[str] = []
        con_data = duckdb.connect(self.db_path, read_only=True)
        try:
            total_tables = self._count_table_texts(con_data)
            with tqdm(total=total_tables, desc="Encoding anchor tables", unit="tables") as pbar:
                for batch_ids, batch_texts in self._iter_table_texts(con_data, batch_size):
                    if not batch_ids:
                        continue
                    embs = model.encode(
                        batch_texts,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        batch_size=len(batch_texts),
                    )
                    if embs is not None and getattr(embs, "size", 0) > 0:
                        all_embs.append(np.asarray(embs, dtype=np.float32))
                        ids.extend(batch_ids)
                    pbar.update(len(batch_ids))
        finally:
            con_data.close()

        if not all_embs:
            raise RuntimeError("No table embeddings generated. Check DuckDB path, resources, and relationship parquet.")

        embs_array = np.vstack(all_embs).astype(np.float32, copy=False)
        np.savez_compressed(self.output_npz, embeddings=embs_array, ids=np.array(ids, dtype=str))
        print(f"Saved anchor-table embeddings: {self.output_npz}, shape={embs_array.shape}, n_ids={len(ids)}")
