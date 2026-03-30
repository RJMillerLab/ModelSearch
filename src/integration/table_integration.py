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
from typing import Any, Dict, List, Optional

import pandas as pd

from src.integration.quick_aug_recognition import KeywordRecognizer


class TableIntegrater:
    def __init__(self, temp_dir: Optional[str] = None, reorder_columns: bool = False):
        self.temp_dir = os.path.abspath(temp_dir) if temp_dir else None
        self.reorder_columns = bool(reorder_columns)
        self.keyword_recognizer = KeywordRecognizer(verbose=False)
        self.output_aligner = KeywordRecognizer(verbose=False)

    def load_tables(self, table_paths: List[str]) -> List[pd.DataFrame]:
        return [pd.read_csv(path) for path in table_paths]

    def _make_columns_unique(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df.columns) == 0:
            return df
        seen = {}
        new_cols: List[str] = []
        changed = False
        for idx, col in enumerate(df.columns):
            base = str(col).strip()
            if not base or base.lower() == "nan":
                base = f"col_{idx}"
            count = seen.get(base, 0)
            if count == 0:
                new_cols.append(base)
            else:
                changed = True
                new_cols.append(f"{base}__{count}")
            seen[base] = count + 1
        if not changed and list(df.columns) == new_cols:
            return df
        out = df.copy()
        out.columns = new_cols
        return out

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
        return self._make_columns_unique(transposed)

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

    def _get_temp_dir(self, temp_dir: Optional[str] = None) -> str:
        target_dir = os.path.abspath(temp_dir or self.temp_dir or "tmp")
        os.makedirs(target_dir, exist_ok=True)
        return target_dir

    def _write_temp_table(self, df: pd.DataFrame, path: str) -> str:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        safe_df = self._make_columns_unique(df)
        safe_df.to_csv(path, index=False)
        return path

    def _prepare_table_for_orientation(
        self,
        *,
        table_df: pd.DataFrame,
        source_path: str,
        orientation: str,
        step_idx: int,
        side: str,
        temp_dir: Optional[str] = None,
    ) -> tuple[pd.DataFrame, str]:
        target_dir = self._get_temp_dir(temp_dir)
        mode = str(orientation or "ori").strip().lower()
        if mode != "tr":
            print(f"[quick_aug] step={step_idx} keep {side} {os.path.basename(source_path)}", flush=True)
            return table_df, source_path

        transposed = self._transpose_promote_first_row(table_df)
        #transposed = self._preprocess_transposed_table(transposed)
        tmp_path = os.path.join(target_dir, f"tmp_step_{step_idx}_{side}.csv")
        self._write_temp_table(transposed, tmp_path)
        print(
            f"[quick_aug] step={step_idx} transpose {side} {os.path.basename(source_path)} -> {tmp_path}",
            flush=True,
        )
        return transposed, tmp_path

    def _integrate_tables_original_alite(
        self,
        table_paths: List[str],
        temp_dir: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        from src.config import ALITE_INTERNAL_REPO

        if not table_paths:
            print("⚠️  ALITE requires file paths, not DataFrames", flush=True)
            return None

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

    def _integrate_tables_alite(
            self,
            table_paths: List[str],
            temp_dir: Optional[str] = None,
        ) -> Optional[pd.DataFrame]:
        if not table_paths:
            print("⚠️  ALITE requires file paths, not DataFrames", flush=True)
            return None

        if len(table_paths) == 1:
            return pd.read_csv(table_paths[0])

        target_dir = self._get_temp_dir(temp_dir)
        current_path = table_paths[0]
        current_df = pd.read_csv(current_path)

        for step_idx, candidate_path in enumerate(table_paths[1:], start=2):
            candidate_df = pd.read_csv(candidate_path)
            candidate_orientation = self.keyword_recognizer.recognize_one_dataframe(
                query_df=current_df,
                retrieved_df=candidate_df,
                table_name=os.path.basename(candidate_path),
            )
            prepared_current_path = current_path
            prepared_candidate_df, prepared_candidate_path = self._prepare_table_for_orientation(
                table_df=candidate_df,
                source_path=candidate_path,
                orientation=candidate_orientation,
                step_idx=step_idx,
                side="right",
                temp_dir=target_dir,
            )
            print(
                f"[alite] iterative step={step_idx - 1} pair=({os.path.basename(prepared_current_path)}, {os.path.basename(prepared_candidate_path)})",
                flush=True,
            )
            pair_df = self._integrate_tables_original_alite(
                [prepared_current_path, prepared_candidate_path],
                temp_dir=target_dir,
            )
            if pair_df is None:
                return None
            current_df = self._make_columns_unique(pair_df)
            current_path = os.path.join(target_dir, f"integrated_step_{step_idx - 1}.csv")
            self._write_temp_table(current_df, current_path)

        return current_df

    def _align_output_to_query(self, query_df: pd.DataFrame, output_df: pd.DataFrame) -> pd.DataFrame:
        if query_df is None or output_df is None or output_df.empty:
            return output_df

        decision = self.output_aligner.recognize_one_dataframe(
            query_df=query_df,
            retrieved_df=output_df,
            table_name="integrated_output",
        )
        if decision != "tr":
            print("[final_align] keep output orientation aligned with query", flush=True)
            return output_df

        print("[final_align] transpose output back to align with query", flush=True)
        return self._make_columns_unique(self._transpose_promote_first_row(output_df))

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
        temp_dir: Optional[str] = None,
        reorder_columns: Optional[bool] = None,
    ) -> Optional[pd.DataFrame]:
        if not table_paths:
            print("❌ No tables to integrate", flush=True)
            return None

        if mode != "alite":
            raise ValueError(f"Only 'alite' integration is supported now, got: {mode}")
        query_df = pd.read_csv(table_paths[0])
        df = self._integrate_tables_alite(table_paths, temp_dir=temp_dir)
        if df is not None:
            df = self._align_output_to_query(query_df, df)
        should_reorder = self.reorder_columns if reorder_columns is None else bool(reorder_columns)
        if df is None:
            return None
        return self._reorder_columns_deterministic(df) if should_reorder else df

    def save_table(self, df: pd.DataFrame, path: str):
        if df is None:
            raise ValueError("Cannot save None DataFrame")
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        df.to_csv(path, index=False)


class GroupTableIntegrater:
    def __init__(self, integrater: Optional[TableIntegrater] = None):
        self.integrater = integrater or TableIntegrater()

    def run(
        self,
        query_to_retrieved_tables: Dict[str, List[str]],
        *,
        mode: str = "alite",
        k: Optional[int] = None,
        temp_dir: Optional[str] = None,
        reorder_columns: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        limit = None if k is None else max(0, int(k))
        for query_table_path, retrieved_table_paths in query_to_retrieved_tables.items():
            query_path = str(query_table_path).strip()
            if not query_path:
                continue
            retrieved_paths = [str(path).strip() for path in (retrieved_table_paths or []) if str(path).strip()]
            if limit is not None:
                retrieved_paths = retrieved_paths[:limit]
            integration_input_table_paths = [query_path] + retrieved_paths
            integrated_df = self.integrater.run(
                integration_input_table_paths,
                mode=mode,
                temp_dir=temp_dir,
                reorder_columns=reorder_columns,
            )
            if integrated_df is None:
                raise RuntimeError(f"Integration returned None for query table: {query_path}")
            results.append(
                {
                    "query_table_path": query_path,
                    "retrieved_table_paths": retrieved_paths,
                    "integration_input_table_paths": integration_input_table_paths,
                    "integrated_df": integrated_df,
                }
            )
        return results

    def save(
        self,
        grouped_results: List[Dict[str, Any]],
        *,
        output_dir: str,
    ) -> List[Dict[str, Any]]:
        saved_results: List[Dict[str, Any]] = []
        target_dir = os.path.abspath(output_dir)
        os.makedirs(target_dir, exist_ok=True)
        for idx, result in enumerate(grouped_results, start=1):
            query_table_path = str(result["query_table_path"]).strip()
            basename = os.path.basename(query_table_path) or f"query_table_{idx:02d}.csv"
            output_name = f"group_{idx:02d}_{basename}"
            output_path = os.path.join(target_dir, output_name)
            self.integrater.save_table(result["integrated_df"], output_path)
            saved = dict(result)
            saved["saved_path"] = output_path
            saved_results.append(saved)
        return saved_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Integrate tables from CSV paths.")
    parser.add_argument("--tables", nargs="+", help="CSV paths to integrate.")
    parser.add_argument("--mode", choices=["alite"], default="alite")
    parser.add_argument("--output_csv", default="tmp/integrated_table.csv")
    parser.add_argument("--temp_dir", default="tmp")
    parser.add_argument("--no_reorder", action="store_true", help="Disable deterministic output-column reordering.")
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

    print(f"[table_integration] mode={args.mode} tables={len(table_paths)} output={os.path.abspath(args.output_csv)}", flush=True)
    integrater = TableIntegrater(temp_dir=args.temp_dir, reorder_columns=not args.no_reorder)
    df = integrater.run(
        table_paths,
        mode=args.mode,
        temp_dir=args.temp_dir,
        reorder_columns=not args.no_reorder,
    )
    if df is None:
        raise RuntimeError("Integration returned None")
    integrater.save_table(df, args.output_csv)
    print(f"Saved integrated table to {os.path.abspath(args.output_csv)}", flush=True)


if __name__ == "__main__":
    main()
