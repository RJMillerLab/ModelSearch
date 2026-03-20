"""
Table to Table Search (Blend_internal) via subprocess.

Why subprocess?
- Blend_internal lives in another repo and also uses a top-level package named `src`.
- Importing Blend_internal classes in-process can cause `src` namespace conflicts.
- Using a subprocess isolates sys.path / sys.modules so Blend_internal can run safely.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
from typing import Any, Iterable, List, Optional

import pandas as pd

from src.config import MODELLAKE_DB, TAB2TAB_OUTPUT_JSON, BLEND_INTERNAL_REPO, INDEX_TABLE

# Guard Blend_internal config.ini mutation (it is shared on disk).
_blend_subprocess_lock = threading.Lock()


def _blend_file_lock_path() -> str:
    """
    Cross-process lock file path for Blend_internal config.ini mutation.

    We need an inter-process lock because backend runs multiple card2tab2card
    search types concurrently, and each one spawns its own Python subprocess
    that calls Blend_internal (another repo) and writes shared config.ini.
    """
    lock_dir = os.path.join(BLEND_INTERNAL_REPO or "", "config")
    if not lock_dir:
        # Fall back to repo-local temp (should never happen if config is valid).
        lock_dir = tempfile.gettempdir()
    os.makedirs(lock_dir, exist_ok=True)
    return os.path.join(lock_dir, ".tab2tab_filelock")


class _BlendFileLock:
    def __enter__(self):
        # Only used on Unix-like platforms (macOS/Linux).
        import fcntl

        self._lock_fh = open(_blend_file_lock_path(), "w", encoding="utf-8")
        fcntl.flock(self._lock_fh, fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc, tb):
        import fcntl

        try:
            fcntl.flock(self._lock_fh, fcntl.LOCK_UN)
        finally:
            try:
                self._lock_fh.close()
            except Exception:
                pass


def _blend_module() -> str:
    """
    Module path used by build_index.md examples.
    Assumes Blend_internal is available under `others/Blend_internal` as a package.
    """
    return "others.Blend_internal.scripts.tab2tab"


def _ensure_blend_exists() -> None:
    if not BLEND_INTERNAL_REPO or not os.path.exists(BLEND_INTERNAL_REPO):
        raise FileNotFoundError(
            "Blend_internal not found. Please clone it into `others/Blend_internal` "
            f"(expected at: {BLEND_INTERNAL_REPO})."
        )


def _run_blend_tab2tab_subprocess(
    *,
    search_type: str,
    query: Any,
    k: int,
    db_path: Optional[str] = None,
    output_json: Optional[str] = None,
    parse_payload: bool = True,
) -> List[int]:
    """
    Run Blend_internal tab2tab script in an isolated subprocess and return [tableid, ...].
    """
    _ensure_blend_exists()

    db_path_use = db_path or MODELLAKE_DB

    # Use caller-provided output path (deterministic) or a temp file (isolated).
    out_path: str
    if output_json:
        out_path = output_json
        parent = os.path.dirname(os.path.abspath(out_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
    else:
        out_fd, out_path = tempfile.mkstemp(prefix="tab2tab_", suffix=".json")
        os.close(out_fd)

    # For multi_column/unionable, query must be a CSV path.
    temp_query_csv_path: Optional[str] = None

    try:
        if search_type in ("single_column", "keyword"):
            if not isinstance(query, (list, tuple, pd.Series)):
                raise ValueError(f"For {search_type}, query must be an iterable of strings.")
            query_str = ",".join(str(x) for x in query)
            query_arg = query_str
        elif search_type in ("multi_column", "unionable"):
            if not isinstance(query, pd.DataFrame):
                raise ValueError(f"For {search_type}, query must be a pandas DataFrame.")
            tmp_dir = tempfile.mkdtemp(prefix="tab2tab_query_")
            temp_query_csv_path = os.path.join(tmp_dir, "query.csv")
            query.to_csv(temp_query_csv_path, index=False, encoding="utf-8")
            query_arg = temp_query_csv_path
        else:
            raise ValueError(f"Unknown search_type: {search_type}")

        cmd = [
            "python",
            "-m",
            _blend_module(),
            "--db_path",
            db_path_use,
            "--output_json",
            out_path,
            "--search_type",
            search_type,
            "--query",
            str(query_arg),
            "--k",
            str(k),
        ]

        # Serialize to avoid concurrent writes to Blend_internal config.ini.
        # This must be a cross-process lock, not only a thread lock.
        with _blend_subprocess_lock:
            with _BlendFileLock():
                proc = subprocess.run(
                    cmd,
                    cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
                    capture_output=True,
                    text=True,
                )

        if proc.returncode != 0:
            raise RuntimeError(
                "Blend_internal tab2tab failed.\n"
                f"cmd: {' '.join(cmd)}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}\n"
            )

        if not parse_payload:
            # Caller only wants the json file written by Blend_internal.
            return []

        with open(out_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        # Blend_internal might output either:
        # - dict: {"results": [tableid,...]} or similar
        # - list: [tableid,...] or [filename,...]
        if isinstance(payload, dict):
            if "results" in payload and isinstance(payload["results"], list):
                return [int(x) for x in payload["results"]]
            if "table_ids" in payload and isinstance(payload["table_ids"], list):
                return [int(x) for x in payload["table_ids"]]
            if "tableid" in payload and isinstance(payload["tableid"], list):
                return [int(x) for x in payload["tableid"]]

        if isinstance(payload, list):
            # Case A: list of ints (or int-like strings)
            if all(isinstance(x, (int, float)) for x in payload):
                return [int(x) for x in payload]
            if all(isinstance(x, str) and x.strip().isdigit() for x in payload):
                return [int(str(x).strip()) for x in payload]

            # Case B: list of filenames / basenames -> map to tableid via DuckDB
            filenames = [os.path.basename(str(x).strip()) for x in payload if str(x).strip()]
            if not filenames:
                return []

            import duckdb

            conn = duckdb.connect(MODELLAKE_DB, read_only=True)
            try:
                placeholders = ",".join(["?"] * len(filenames))
                query_sql = f"""
                    SELECT DISTINCT tableid, filename
                    FROM {INDEX_TABLE}
                    WHERE rowid = -1 AND filename IN ({placeholders})
                """
                rows = conn.execute(query_sql, filenames).fetchall()
            finally:
                conn.close()

            filename_to_tid = {os.path.basename(str(fname)).strip(): int(tid) for tid, fname in rows}
            out: List[int] = []
            seen = set()
            for fn in filenames:
                tid = filename_to_tid.get(fn)
                if tid is None:
                    continue
                if tid not in seen:
                    seen.add(tid)
                    out.append(tid)
            return out

        raise ValueError(f"Unexpected Blend_internal output JSON format: {payload!r}")
    finally:
        try:
            if not output_json and os.path.exists(out_path):
                os.remove(out_path)
        except Exception:
            pass
        if temp_query_csv_path:
            try:
                # temp_query_csv_path is inside a tmp dir; remove best-effort.
                tmp_dir = os.path.dirname(temp_query_csv_path)
                if os.path.isdir(tmp_dir):
                    for fn in os.listdir(tmp_dir):
                        try:
                            os.remove(os.path.join(tmp_dir, fn))
                        except Exception:
                            pass
                    os.rmdir(tmp_dir)
            except Exception:
                pass


def search_single_column(query_values: Iterable[Any], k: int = 10, db_path: Optional[str] = None) -> List[int]:
    return _run_blend_tab2tab_subprocess(search_type="single_column", query=list(query_values), k=k, db_path=db_path)


def search_keyword(query_values: List[str], k: int = 10, db_path: Optional[str] = None) -> List[int]:
    return _run_blend_tab2tab_subprocess(search_type="keyword", query=list(query_values), k=k, db_path=db_path)


def search_multi_column(query_dataset: pd.DataFrame, k: int = 10, db_path: Optional[str] = None) -> List[int]:
    return _run_blend_tab2tab_subprocess(search_type="multi_column", query=query_dataset, k=k, db_path=db_path)


def search_unionable(query_dataset: pd.DataFrame, k: int = 10, db_path: Optional[str] = None) -> List[int]:
    return _run_blend_tab2tab_subprocess(search_type="unionable", query=query_dataset, k=k, db_path=db_path)


def search_table2table(
    query: Any,
    search_type: str = "single_column",
    k: int = 10,
    db_path: Optional[str] = None,
    output_json: Optional[str] = None,
) -> List[int]:
    """
    Unified interface for the 4 supported table-to-table search types.
    This mirrors the old in-process wrapper API but uses subprocess.
    """
    if search_type == "single_column":
        if not isinstance(query, (list, tuple, pd.Series)):
            raise ValueError("For single_column search, query must be an iterable of values")
        return _run_blend_tab2tab_subprocess(
            search_type="single_column",
            query=list(query),
            k=k,
            db_path=db_path,
            output_json=output_json,
        )
    if search_type == "multi_column":
        if not isinstance(query, pd.DataFrame):
            raise ValueError("For multi_column search, query must be a pandas DataFrame")
        return _run_blend_tab2tab_subprocess(
            search_type="multi_column",
            query=query,
            k=k,
            db_path=db_path,
            output_json=output_json,
        )
    if search_type == "keyword":
        if not isinstance(query, list) or not all(isinstance(x, str) for x in query):
            raise ValueError("For keyword search, query must be a list of strings")
        return _run_blend_tab2tab_subprocess(
            search_type="keyword",
            query=list(query),
            k=k,
            db_path=db_path,
            output_json=output_json,
        )
    if search_type == "unionable":
        if not isinstance(query, pd.DataFrame):
            raise ValueError("For unionable search, query must be a pandas DataFrame")
        return _run_blend_tab2tab_subprocess(
            search_type="unionable",
            query=query,
            k=k,
            db_path=db_path,
            output_json=output_json,
        )
    raise ValueError("Unknown search_type: must be 'single_column', 'multi_column', 'keyword', or 'unionable'")


def main() -> None:
    """
    Minimal CLI for local testing (kept consistent with build_index.md usage).
    """
    import argparse
    start = None
    parser = argparse.ArgumentParser(description="Table to Table Search using Blend_internal (subprocess wrapper)")
    parser.add_argument("--search_type", choices=["single_column", "multi_column", "keyword", "unionable"], required=True)
    parser.add_argument("--query", default=None, required=True, help="CSV path for multi_column/unionable, comma-separated values for others.")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output_json", default=None, help="Optional output json path to write Blend_internal results.")
    args = parser.parse_args()

    if args.search_type in ("single_column", "keyword"):
        query: Any = [x.strip() for x in str(args.query).split(",") if x.strip()]
    else:
        query = pd.read_csv(args.query)

    start = __import__("time").time()
    # If caller provides --output_json, Blend_internal will already write it.
    # In that case we skip parsing payload to avoid extra work.
    parse_payload = args.output_json is None
    results = _run_blend_tab2tab_subprocess(
        search_type=args.search_type,
        query=query,
        k=args.k,
        output_json=args.output_json,
        parse_payload=parse_payload,
    )
    if args.output_json:
        print(f"Saved json: {args.output_json}")
    else:
        print(f"Found {len(results)} tables: {results}")
    print(f"Total time: {__import__('time').time() - start:.2f}s")


if __name__ == "__main__":
    main()

