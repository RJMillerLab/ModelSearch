#!/usr/bin/env python3
"""Keyword-based orientation recognition: decide whether a retrieved table should stay ori or become tr."""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Iterable, List, Sequence, Set, Tuple

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.config import TABLE_BASE_DIRS


def _normalize_text(value: object) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def _unique_non_empty(values: Iterable[object]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for value in values:
        norm = _normalize_text(value)
        if not norm or norm == "nan" or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def _resolve_table_path(path_or_name: str) -> str:
    raw = str(path_or_name or "").strip()
    if not raw:
        raise ValueError("table path is empty")
    if os.path.isfile(raw):
        return os.path.abspath(raw)
    basename = os.path.basename(raw)
    for base_dir in TABLE_BASE_DIRS:
        candidate = os.path.join(str(base_dir), basename)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(f"Cannot resolve table path: {path_or_name}")

def _extract_axis_keywords(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    header_keywords = _unique_non_empty(list(df.columns))
    first_col_values: Sequence[object] = []
    if df.shape[1] > 0:
        first_col_values = list(df.iloc[:, 0].tolist())
    first_col_keywords = _unique_non_empty(first_col_values)
    return header_keywords, first_col_keywords


class BaseKeywordRecognizer:
    def __init__(self, *, verbose: bool = False):
        self.verbose = bool(verbose)

    def _overlap(self, left: Sequence[str], right: Sequence[str]) -> List[str]:
        return sorted(set(left) & set(right))

    def _preview(self, values: Sequence[str], *, limit: int = 6) -> str:
        if not values:
            return "none"
        text = ", ".join(values[:limit])
        if len(values) > limit:
            text += ", ..."
        return text

    def _matrix_text(
        self,
        *,
        hh: Sequence[str],
        hc: Sequence[str],
        ch: Sequence[str],
        cc: Sequence[str],
    ) -> str:
        return (
            "overlap_matrix:\n"
            f"  q.header x t.header = {len(hh)} [{self._preview(hh)}]\n"
            f"  q.header x t.col1   = {len(hc)} [{self._preview(hc)}]\n"
            f"  q.col1   x t.header = {len(ch)} [{self._preview(ch)}]\n"
            f"  q.col1   x t.col1   = {len(cc)} [{self._preview(cc)}]"
        )


class KeywordRecognizer(BaseKeywordRecognizer):
    """Recognize `ori` vs `tr` with header/first-column keyword overlap."""

    def recognize_one_dataframe(self, *, query_df: pd.DataFrame, retrieved_df: pd.DataFrame, table_name: str = "") -> str:
        query_header_keywords, query_first_col_keywords = _extract_axis_keywords(query_df)
        cand_header_keywords, cand_first_col_keywords = _extract_axis_keywords(retrieved_df)

        same_header = self._overlap(query_header_keywords, cand_header_keywords)
        same_first_col = self._overlap(query_first_col_keywords, cand_first_col_keywords)
        cross_header_to_col1 = self._overlap(query_header_keywords, cand_first_col_keywords)
        cross_col1_to_header = self._overlap(query_first_col_keywords, cand_header_keywords)

        keep_score = len(same_header) + len(same_first_col)
        transpose_score = len(cross_header_to_col1) + len(cross_col1_to_header)
        result = "tr" if transpose_score > keep_score else "ori"

        if self.verbose:
            action = "transpose" if result == "tr" else "no transpose"
            reason = (
                f"cross overlap ({transpose_score}) > diagonal overlap ({keep_score})"
                if result == "tr"
                else f"diagonal overlap ({keep_score}) >= cross overlap ({transpose_score})"
            )
            print(
                f"{table_name or '(table)'}: {action}\n"
                f"{self._matrix_text(hh=same_header, hc=cross_header_to_col1, ch=cross_col1_to_header, cc=same_first_col)}\n"
                f"reason: {reason}",
                flush=True,
            )

        return result

    def recognize_dataframes(
        self,
        *,
        query_df: pd.DataFrame,
        retrieved_dfs: Sequence[pd.DataFrame],
        table_names: Sequence[str] | None = None,
    ) -> List[str]:
        outputs: List[str] = []
        names = list(table_names) if table_names is not None else []

        for idx, retrieved_df in enumerate(retrieved_dfs):
            table_name = names[idx] if idx < len(names) else ""
            outputs.append(
                self.recognize_one_dataframe(
                    query_df=query_df,
                    retrieved_df=retrieved_df,
                    table_name=table_name,
                )
            )
        return outputs

    def recognize_paths(self, *, query_table: str, retrieved_tables: Sequence[str]) -> List[str]:
        query_path = _resolve_table_path(query_table)
        query_df = pd.read_csv(query_path)
        retrieved_dfs: List[pd.DataFrame] = []
        table_names: List[str] = []

        for raw_path in retrieved_tables:
            source_path = _resolve_table_path(raw_path)
            retrieved_dfs.append(pd.read_csv(source_path))
            table_names.append(os.path.basename(source_path))

        return self.recognize_dataframes(
            query_df=query_df,
            retrieved_dfs=retrieved_dfs,
            table_names=table_names,
        )

    def recognize(self, *, query_table: str, retrieved_tables: Sequence[str]) -> List[str]:
        return self.recognize_paths(query_table=query_table, retrieved_tables=retrieved_tables)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recognize whether each retrieved table should be ori or tr.")
    parser.add_argument("--query_table", required=True, help="Query table path or basename.")
    parser.add_argument("--retrieved_tables", nargs="+", required=True, help="Retrieved table paths or basenames.")
    parser.add_argument("--verbose", action="store_true", help="Print per-table overlap details.")
    args = parser.parse_args()

    recognizer = KeywordRecognizer(verbose=args.verbose)
    results = recognizer.recognize(query_table=args.query_table, retrieved_tables=args.retrieved_tables)
    print(results, flush=True)


if __name__ == "__main__":
    main()
