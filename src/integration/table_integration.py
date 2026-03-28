"""
Table Integration Implementation

Integrates multiple tables using various methods:
- Union: Combine all rows from all tables
- Intersection: Find common rows across all tables
- ALITE: FD-based integration using dialite_internal (requires dialite_internal repository)
- Outer Join: Merge all tables using outer join
"""

import os
import sys
import io
from contextlib import redirect_stdout, redirect_stderr
import pandas as pd
from typing import List, Optional

class TableIntegrater:
    def __init__(self):
        pass
    
    def load_tables(self, table_paths: List[str]) -> List[pd.DataFrame]:
        return [pd.read_csv(path) for path in table_paths]

    def _integrate_tables_union(self, table_paths: List[str]) -> Optional[pd.DataFrame]:
        """
        Integrate multiple tables using Union operation.
        Returns the full integrated result (no row limit).
        
        Args:
            tables: List of DataFrames to integrate
            
        Returns:
            Integrated DataFrame or None if integration fails
        """
        if not table_paths or len(table_paths) == 0:
            return None
        
        tables = self.load_tables(table_paths)
        if len(tables) == 1:
            return tables[0].copy()
        
        # Use pandas concat for union (combine all rows)
        all_columns: List[str] = []
        for df in tables:
            for col in df.columns:
                if col not in all_columns:
                    all_columns.append(col)
        aligned_tables = [df.reindex(columns=all_columns) for df in tables]
        integrated = pd.concat(aligned_tables, axis=0, ignore_index=True)
        integrated = integrated.drop_duplicates()
        return integrated

    def _integrate_tables_intersection(self, table_paths: List[str]) -> Optional[pd.DataFrame]:
        """
        Integrate multiple tables using Intersection operation (find common rows).
        Returns the full result (no row limit).
        
        Args:
            tables: List of DataFrames to integrate
            
        Returns:
            Integrated DataFrame with common rows or None if no common rows
        """
        if not table_paths or len(table_paths) == 0:
            return None
        
        tables = self.load_tables(table_paths)
        
        if len(tables) == 1:
            return tables[0].copy()

        # Find common columns
        common_columns = set(tables[0].columns)
        for df in tables[1:]:
            common_columns = common_columns.intersection(set(df.columns))

        if not common_columns:
            print("⚠️  No common columns found for intersection")
            return pd.DataFrame()

        # Convert to string for comparison
        common_columns = list(common_columns)

        # Find intersection (rows that appear in all tables)
        result = tables[0][common_columns].copy()
        result['_temp_key'] = result.apply(lambda x: '|'.join(x.astype(str)), axis=1)

        for df in tables[1:]:
            df_subset = df[common_columns].copy()
            df_subset['_temp_key'] = df_subset.apply(lambda x: '|'.join(x.astype(str)), axis=1)
            result = result[result['_temp_key'].isin(df_subset['_temp_key'])]

        result = result[common_columns]
        return result

    def _integrate_tables_alite(self, table_paths: List[str]) -> Optional[pd.DataFrame]:
        """
        Integrate tables using ALITE FD-based algorithm.
        Returns the full result (no row limit).
        
        Args:
            tables: List of DataFrames (not used directly, but kept for consistency)
            table_paths: List of paths to CSV files (required for ALITE)
            
        Returns:
            Integrated DataFrame or None if integration fails
        """
        from src.config import DIALITE_INTERNAL_REPO
        dialite_repo = DIALITE_INTERNAL_REPO
        if dialite_repo not in sys.path:
            sys.path.insert(0, dialite_repo)

        import alite.alite_fd as alite_module

        if not table_paths:
            print("⚠️  ALITE requires file paths, not DataFrames")
            return None

        alite_verbose = os.environ.get("ALITE_VERBOSE", "").strip().lower() in {"1", "true", "yes", "on"}
        if alite_verbose:
            result_FD, stats_df, debug_dict = alite_module.FDAlgorithm(table_paths.copy())
        else:
            # ALITE emits many internal progress prints; suppress them by default.
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result_FD, stats_df, debug_dict = alite_module.FDAlgorithm(table_paths.copy())
        if result_FD is not None and len(result_FD) > 0:
            return result_FD
        return result_FD

    def _integrate_tables_outer_join(self, table_paths: List[str]) -> Optional[pd.DataFrame]:
        """
        Integrate tables using outer join (merge all tables on index).
        Returns the full result (no row limit).
        
        Args:
            tables: List of DataFrames to integrate
            
        Returns:
            Integrated DataFrame or None if integration fails
        """
        if not table_paths or len(table_paths) == 0:
            return None
        
        tables = self.load_tables(table_paths)
        
        if len(tables) == 1:
            return tables[0].copy()
        
        # Start with first table; merge others with outer join on index
        result = tables[0].copy()
        for df in tables[1:]:
            result_reset = result.reset_index(drop=True)
            df_reset = df.reset_index(drop=True)
            result = pd.concat([result_reset, df_reset], axis=1, join='outer')
        return result
    
    def _reorder_columns_deterministic(self, df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """Deterministically reorder columns for readability/comparability.

        Rules (table-only; no external reference):
        1) Columns that are not entirely null/empty first.
        2) Within those, higher non-null rate first.
        3) Columns that are entirely null/empty always go to the end.
        4) Stable tie-breaker: original column order.
        """
        if df is None or df.empty or len(df.columns) == 0:
            return df
        cols = list(df.columns)
        mask = df.notna() & (df != "")
        rate = mask.mean().values
        is_all_null = (rate == 0).astype(int)
        order = sorted(range(len(cols)), key=lambda i: (is_all_null[i], -rate[i], i))
        ordered_cols = [cols[i] for i in order]
        if ordered_cols != cols:
            if verbose:
                print(f"[reorder] columns changed\n  before: {cols}\n  after:  {ordered_cols}")
            else:
                print(f"[reorder] columns changed")
            return df[ordered_cols]
        return df
    
    def run(self, table_paths, mode: str = "union") -> Optional[pd.DataFrame]:
        if not table_paths:
            print("❌ No tables to integrate")
            return None

        if mode == "union":
            df = self._integrate_tables_union(table_paths)
        elif mode == "intersection":
            df = self._integrate_tables_intersection(table_paths)
        elif mode == "outer_join":
            df = self._integrate_tables_outer_join(table_paths)
        elif mode == "alite":
            df = self._integrate_tables_alite(table_paths)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        df = self._reorder_columns_deterministic(df) if df is not None else None
        return df
    
    def save_table(self, df: pd.DataFrame, path: str):
        df.to_csv(path, index=False)

if __name__ == "__main__":
    table_paths = [
        "../ModelTables/data/processed/deduped_hugging_csvs_v2_251117/0a0af57dcb_table1.csv",
        "../ModelTables/data/processed/deduped_hugging_csvs_v2_251117/0a0c8a25be_table2.csv",
        "../ModelTables/data/processed/deduped_hugging_csvs_v2_251117/0a0d2a99ba_table1.csv",
    ]
    table_integrater = TableIntegrater()
    df = table_integrater.run(table_paths, mode="union")
    table_integrater.save_table(df, "tmp/integrated_table.csv")
    print('Saved integrated table to tmp/integrated_table.csv')
