"""
Metric-based evaluation utilities (no LLM) for ModelSearch.

This module provides simple, IR-inspired diversity / novelty metrics that work in
two settings:

- Model card search results: ranked lists of model_ids (uses existing
  `data/card2card_embeddings.npz` to avoid re-encoding text).
- Integrated tables: pandas DataFrames from table search / model search
  integration (treats rows as "items" and embeds short row text with a
  SentenceTransformer).

Design goals:
- No reliance on ground-truth labels.
- Same family of metrics (intra-list diversity, sequential novelty) for both
  model-card rankings and integrated tables, so the numbers are comparable
  across the two pipelines at least qualitatively.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore


def _cosine_sim_matrix(emb: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix for embeddings of shape (n, d)."""
    if emb.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {emb.shape}")
    # Normalize each row to unit length
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_norm = emb / norms
    return emb_norm @ emb_norm.T


def intra_list_diversity(emb: np.ndarray) -> float:
    """
    Intra-list diversity (ILD): average pairwise dissimilarity 1 - cos_sim.

    Args:
        emb: Array of shape (n_items, dim)
    """
    n = emb.shape[0]
    if n <= 1:
        return 0.0
    sim = _cosine_sim_matrix(emb)
    # Use upper triangle without diagonal
    i_upper = np.triu_indices(n, k=1)
    dissimilar = 1.0 - sim[i_upper]
    return float(dissimilar.mean()) if dissimilar.size else 0.0


def sequential_novelty(emb: np.ndarray) -> float:
    """
    Sequential novelty: for each position i>0, 1 - max_j<i cos_sim(i, j).

    Higher means later items tend to introduce new information instead of
    duplicating earlier ones.

    Args:
        emb: Array of shape (n_items, dim), in ranked order.
    """
    n = emb.shape[0]
    if n <= 1:
        return 0.0
    sim = _cosine_sim_matrix(emb)
    novelties: List[float] = []
    for i in range(1, n):
        max_prev = float(sim[i, :i].max())
        novelties.append(1.0 - max_prev)
    return float(np.mean(novelties)) if novelties else 0.0


# ---------------------------------------------------------------------------
# Model-card ranking diversity (uses precomputed embeddings)
# ---------------------------------------------------------------------------

def load_card2card_embeddings(emb_npz_path: str) -> Tuple[Dict[str, int], np.ndarray]:
    """
    Load card2card embeddings (as built by baseline scripts).

    Returns:
        id_to_idx: mapping model_id -> row index in emb matrix
        emb: np.ndarray of shape (n_models, dim)
    """
    if not os.path.exists(emb_npz_path):
        raise FileNotFoundError(f"Embedding NPZ not found: {emb_npz_path}")
    data = np.load(emb_npz_path)
    ids = data["ids"].tolist()
    emb = data["embeddings"].astype("float32")
    id_to_idx = {str(mid): i for i, mid in enumerate(ids)}
    return id_to_idx, emb


def evaluate_model_list_diversity(
    model_ids: Sequence[str],
    emb_npz_path: str = "data/card2card_embeddings.npz",
    max_items: int = 50,
) -> Dict[str, float]:
    """
    Compute diversity metrics for a ranked list of model_ids.

    This is intended for:
    - Card2Card results (model search)
    - Card2Tab2Card results (table search → model cards)

    Args:
        model_ids: Ranked list of model IDs (strings).
        emb_npz_path: Path to card2card embedding NPZ (ids + embeddings).
        max_items: Cap on how many top items to use for metrics (for consistency).
    """
    id_to_idx, emb_all = load_card2card_embeddings(emb_npz_path)
    chosen_indices: List[int] = []
    for mid in model_ids:
        if len(chosen_indices) >= max_items:
            break
        idx = id_to_idx.get(str(mid))
        if idx is not None:
            chosen_indices.append(idx)
    if not chosen_indices:
        return {
            "num_items": 0.0,
            "intra_list_diversity": 0.0,
            "sequential_novelty": 0.0,
        }
    emb = emb_all[chosen_indices]
    return {
        "num_items": float(len(chosen_indices)),
        "intra_list_diversity": intra_list_diversity(emb),
        "sequential_novelty": sequential_novelty(emb),
    }


def compare_model_lists_diversity(
    list_a: Sequence[str],
    list_b: Sequence[str],
    emb_npz_path: str = "data/card2card_embeddings.npz",
    max_items: int = 50,
) -> Dict[str, Dict[str, float]]:
    """
    Compare two model-card rankings using diversity metrics.

    Returns:
        {
          "list_a": {...metrics...},
          "list_b": {...metrics...},
          "overlap": {
              "jaccard": ...,
              "intersection_size": ...,
              "union_size": ...
          }
        }
    """
    metrics_a = evaluate_model_list_diversity(list_a, emb_npz_path=emb_npz_path, max_items=max_items)
    metrics_b = evaluate_model_list_diversity(list_b, emb_npz_path=emb_npz_path, max_items=max_items)

    set_a = set(str(x) for x in list_a[:max_items])
    set_b = set(str(x) for x in list_b[:max_items])
    inter = set_a & set_b
    union = set_a | set_b
    jacc = float(len(inter) / len(union)) if union else 0.0

    overlap = {
        "jaccard": jacc,
        "intersection_size": float(len(inter)),
        "union_size": float(len(union)),
    }
    return {"list_a": metrics_a, "list_b": metrics_b, "overlap": overlap}


# ---------------------------------------------------------------------------
# Integrated table diversity (row-level, uses SentenceTransformer if available)
# ---------------------------------------------------------------------------

def _row_texts_from_table(
    df: pd.DataFrame,
    max_rows: int = 50,
    max_cols: int = 10,
) -> List[str]:
    """
    Turn table rows into short text snippets for embedding.

    We only need a rough signal of content variety, so we:
    - Use up to max_rows rows
    - Only consider first max_cols columns
    """
    if df is None or df.empty:
        return []
    df_sub = df.iloc[:max_rows, :max_cols]
    texts: List[str] = []
    cols = list(df_sub.columns)
    for _, row in df_sub.iterrows():
        parts: List[str] = []
        for c in cols:
            v = row[c]
            if pd.isna(v):
                continue
            s = str(v).strip()
            if not s:
                continue
            parts.append(f"{c}: {s}")
        if parts:
            texts.append(" | ".join(parts))
    return texts


def _embed_texts(
    texts: Sequence[str],
    model_name: str = "all-MiniLM-L6-v2",
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Embed a list of texts with SentenceTransformer.

    This is only used for table diversity; model-card rankings reuse existing
    FAISS embeddings instead.
    """
    if not texts:
        return np.zeros((0, 1), dtype="float32")
    if SentenceTransformer is None:
        raise RuntimeError(
            "sentence-transformers is not available; install it to use table diversity metrics."
        )
    model = SentenceTransformer(model_name, device=device or "cpu")
    emb = model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
    return emb.astype("float32")


def evaluate_table_diversity(
    df: pd.DataFrame,
    max_rows: int = 50,
    max_cols: int = 10,
    model_name: str = "all-MiniLM-L6-v2",
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute diversity metrics for an integrated table.

    We treat each sampled row as an "item" and apply the same intra-list
    diversity / sequential novelty metrics as for ranked model lists.

    Args:
        df: Integrated table (pandas DataFrame).
        max_rows: Number of rows to sample (top rows).
        max_cols: Number of columns to include in row text.
        model_name: SentenceTransformer model for row embeddings.
        device: "cpu" or "cuda" (optional).
    """
    if df is None or df.empty:
        return {
            "num_rows": 0.0,
            "num_columns": 0.0,
            "avg_unique_values_per_column": 0.0,
            "intra_list_diversity": 0.0,
            "sequential_novelty": 0.0,
        }

    # Basic structural stats
    n_rows = int(len(df))
    n_cols = int(len(df.columns))
    if n_cols > 0:
        unique_counts = [df[c].nunique(dropna=True) for c in df.columns]
        avg_unique = float(np.mean(unique_counts))
    else:
        avg_unique = 0.0

    # Content diversity from row embeddings
    texts = _row_texts_from_table(df, max_rows=max_rows, max_cols=max_cols)
    if not texts:
        ild = 0.0
        nov = 0.0
    else:
        emb = _embed_texts(texts, model_name=model_name, device=device)
        ild = intra_list_diversity(emb)
        nov = sequential_novelty(emb)

    return {
        "num_rows": float(n_rows),
        "num_columns": float(n_cols),
        "avg_unique_values_per_column": avg_unique,
        "intra_list_diversity": ild,
        "sequential_novelty": nov,
    }


def compare_tables_diversity(
    table_a: pd.DataFrame,
    table_b: pd.DataFrame,
    max_rows: int = 50,
    max_cols: int = 10,
    model_name: str = "all-MiniLM-L6-v2",
    device: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compare two integrated tables (e.g., Table Search vs Model Search) using
    the same diversity metrics.

    Returns:
        {
          "table_a": {...metrics...},
          "table_b": {...metrics...}
        }
    """
    metrics_a = evaluate_table_diversity(
        table_a,
        max_rows=max_rows,
        max_cols=max_cols,
        model_name=model_name,
        device=device,
    )
    metrics_b = evaluate_table_diversity(
        table_b,
        max_rows=max_rows,
        max_cols=max_cols,
        model_name=model_name,
        device=device,
    )
    return {"table_a": metrics_a, "table_b": metrics_b}

