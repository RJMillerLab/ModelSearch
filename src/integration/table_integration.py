"""
Table Integration Implementation

Integrates multiple tables using ALITE only.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from typing import List, Optional

import pandas as pd


class TableIntegrater:
    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = os.path.abspath(temp_dir) if temp_dir else None

    def load_tables(self, table_paths: List[str]) -> List[pd.DataFrame]:
        return [pd.read_csv(path) for path in table_paths]

    def _transpose_promote_first_row(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None:
            return pd.DataFrame()
        if df.empty:
            out = df.T.reset_index()
            out.columns = [str(col).strip() or f"col_{idx}" for idx, col in enumerate(out.columns)]
            return out

        transposed = df.T.reset_index()
        new_columns: List[str] = []
        for idx, value in enumerate(transposed.iloc[0].tolist()):
            text = str(value).strip()
            new_columns.append(text if text and text.lower() != "nan" else f"col_{idx}")
        transposed = transposed.iloc[1:].reset_index(drop=True)
        transposed.columns = new_columns
        return transposed

    def _preprocess_transposed_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        After transpose, the top-left header cell can ambiguously act like both
        a former row-label marker and a real column header. Clear it so downstream
        integration is less likely to over-interpret that cell as schema content.
        """
        if df is None or len(df.columns) == 0:
            return df
        out = df.copy()
        cols = list(out.columns)
        cols[0] = ""
        out.columns = cols
        return out

    def _postprocess_table_paths(
        self,
        table_paths: List[str],
        table_orientations: Optional[List[str]] = None,
        temp_dir: Optional[str] = None,
    ) -> List[str]:
        if not table_orientations:
            return list(table_paths)

        assert len(table_orientations) == len(table_paths), (
            f"table_orientations length {len(table_orientations)} must match "
            f"table_paths length {len(table_paths)}"
        )

        target_dir = os.path.abspath(temp_dir or self.temp_dir or "tmp")
        os.makedirs(target_dir, exist_ok=True)
        out_paths: List[str] = []

        for idx, (path, orientation) in enumerate(zip(table_paths, table_orientations), start=1):
            mode = str(orientation or "ori").strip().lower()
            if mode == "ori":
                out_paths.append(path)
                continue
            if mode != "tr":
                raise ValueError(f"Unsupported table orientation: {orientation!r}; expected 'ori' or 'tr'")

            df = pd.read_csv(path)
            transposed = self._transpose_promote_first_row(df)
            transposed = self._preprocess_transposed_table(transposed)
            tmp_path = os.path.join(target_dir, f"tmp_{idx}.csv")
            transposed.to_csv(tmp_path, index=False)
            print(f"[postprocess] transpose table {idx}: {os.path.basename(path)} -> {tmp_path}", flush=True)
            out_paths.append(tmp_path)

        return out_paths

    def _integrate_tables_alite(
        self,
        table_paths: List[str],
        table_orientations: Optional[List[str]] = None,
        temp_dir: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        from src.config import ALITE_INTERNAL_REPO

        if not table_paths:
            print("⚠️  ALITE requires file paths, not DataFrames", flush=True)
            return None

        table_paths = self._postprocess_table_paths(table_paths, table_orientations=table_orientations, temp_dir=temp_dir)
        alite_repo = os.path.abspath(ALITE_INTERNAL_REPO)
        alite_codes_dir = os.path.join(alite_repo, "codes")
        if alite_codes_dir not in sys.path:
            sys.path.insert(0, alite_codes_dir)
        import alite_fd as alite_module

        alite_verbose = os.environ.get("ALITE_VERBOSE", "").strip().lower() in {"1", "true", "yes", "on"}
        print(f"[alite] input_tables={len(table_paths)}", flush=True)
        if alite_verbose:
            result_fd, _stats_df, _debug_dict = alite_module.FDAlgorithm(table_paths.copy(), cluster="__".join(os.path.splitext(os.path.basename(table_path))[0] for table_path in table_paths))
        else:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result_fd, _stats_df, _debug_dict = alite_module.FDAlgorithm(table_paths.copy(), cluster="__".join(os.path.splitext(os.path.basename(table_path))[0] for table_path in table_paths))
        return result_fd

    def _reorder_columns_deterministic(self, df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        if df is None or df.empty or len(df.columns) == 0:
            return df
        cols = list(df.columns)
        mask = df.notna() & (df != "")
        rate = mask.mean().values
        is_all_null = (rate == 0).astype(int)
        order = sorted(range(len(cols)), key=lambda i: (is_all_null[i], -rate[i], i))
        ordered_cols = [cols[i] for i in order]
        if ordered_cols != cols:
            print("[reorder] columns changed" if not verbose else f"[reorder] columns changed\n  before: {cols}\n  after:  {ordered_cols}", flush=True)
            return df[ordered_cols]
        return df

    def run(
        self,
        table_paths: List[str],
        mode: str = "alite",
        table_orientations: Optional[List[str]] = None,
        temp_dir: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        if not table_paths:
            print("❌ No tables to integrate", flush=True)
            return None

        if mode != "alite":
            raise ValueError(f"Only 'alite' integration is supported now, got: {mode}")
        df = self._integrate_tables_alite(table_paths, table_orientations=table_orientations, temp_dir=temp_dir)

        return self._reorder_columns_deterministic(df) if df is not None else None

    def save_table(self, df: pd.DataFrame, path: str):
        if df is None:
            raise ValueError("Cannot save None DataFrame")
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Integrate tables from CSV paths.")
    parser.add_argument("--tables", nargs="+", help="CSV paths to integrate.")
    parser.add_argument("--mode", choices=["alite"], default="alite")
    parser.add_argument("--table_orientations", nargs="*", help="Optional ori/tr list aligned with --tables.")
    parser.add_argument("--output_csv", default="tmp/integrated_table.csv")
    parser.add_argument("--temp_dir", default="tmp")
    args = parser.parse_args()

    if not args.tables:
        print("No --tables provided.", flush=True)
        print(
            "Example:\n"
            "python -m src.integration.table_integration "
            "--tables a.csv b.csv c.csv --mode alite --output_csv tmp/integrated_table.csv",
            flush=True,
        )
        return

    table_paths = [os.path.abspath(p) for p in args.tables]
    missing = [p for p in table_paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"Missing table paths: {missing}")

    if args.table_orientations and len(args.table_orientations) != len(table_paths):
        raise ValueError("--table_orientations length must match --tables length")

    print(f"[table_integration] mode={args.mode} tables={len(table_paths)} output={os.path.abspath(args.output_csv)}", flush=True)
    integrater = TableIntegrater(temp_dir=args.temp_dir)
    df = integrater.run(
        table_paths,
        mode=args.mode,
        table_orientations=list(args.table_orientations) if args.table_orientations else None,
        temp_dir=args.temp_dir,
    )
    if df is None:
        raise RuntimeError("Integration returned None")
    integrater.save_table(df, args.output_csv)
    print(f"Saved integrated table to {os.path.abspath(args.output_csv)}", flush=True)


if __name__ == "__main__":
    main()
