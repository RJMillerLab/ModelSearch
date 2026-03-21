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

#from src.config import MODELLAKE_DB, TAB2TAB_OUTPUT_JSON, BLEND_INTERNAL_REPO, INDEX_TABLE
from src.config import *
from src.utils import resolve_table_path

# Guard Blend_internal config.ini mutation (it is shared on disk).
_blend_subprocess_lock = threading.Lock()


def _extract_keyword_query_from_table(table_path: str) -> List[str]:
    """Build keyword query from table headers."""
    out: List[str] = []
    seen = set()
    try:
        df = pd.read_csv(table_path, nrows=0)
    except Exception:
        return []
    for c in df.columns:
        s = str(c).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _extract_single_column_query_from_table(table_path: str, max_rows_per_table: int = 100) -> List[str]:
    """Build single-column query from first-column values."""
    try:
        df = pd.read_csv(table_path, nrows=max_rows_per_table)
    except Exception:
        return []
    if len(df.columns) == 0:
        return []
    vals = df[df.columns[0]].dropna().astype(str).tolist()
    return [v for v in vals if str(v).strip()]


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


def search_table2table(
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

    # Used for debug prints (keyword/single_column paths).
    query_tokens: List[str] = []
    query_arg_preview: str = ""

    try:
        if not isinstance(query, str):
            raise ValueError("query must be a table name/path string")
        table_path = resolve_table_path(query) or (query if os.path.exists(query) else None)
        if not table_path:
            raise ValueError(f"No existing table found for query: {query!r}")

        if search_type in ("single_column", "keyword"):
            if search_type == "keyword":
                query_tokens = _extract_keyword_query_from_table(table_path)
            else:
                query_tokens = _extract_single_column_query_from_table(table_path)
            if not query_tokens:
                raise ValueError(f"Empty extracted query from table path: {table_path!r}")
            query_arg = ",".join(query_tokens)
            # Keep logs readable; the actual query string can be long.
            query_arg_preview = query_arg[:300] + ("..." if len(query_arg) > 300 else "")
            print(
                f"[tab2tab-debug] search_type={search_type} "
                f"table={os.path.basename(table_path)} "
                f"query_tokens={len(query_tokens)} "
                f"tokens_preview={query_tokens[:20]!r} "
                f"query_arg_preview={query_arg_preview!r}",
                flush=True,
            )
        elif search_type in ("multi_column", "unionable"):
            # Blend expects query csv path for these modes.
            query_arg = table_path
        else:
            raise ValueError(f"Unknown search_type: {search_type}")

        cmd = ["python", "-m", _blend_module(), "--db_path", db_path, "--output_json", out_path, "--search_type", search_type, "--query", str(query_arg), "--k", str(k)]

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
                out_ids = [int(x) for x in payload["results"]]
                if not out_ids and search_type in ("keyword", "single_column"):
                    print(
                        f"[tab2tab-debug] empty ids from Blend. "
                        f"search_type={search_type} query_arg_preview={query_arg_preview!r} "
                        f"tokens_preview={query_tokens[:20]!r}",
                        flush=True,
                    )
                return out_ids
            if "table_ids" in payload and isinstance(payload["table_ids"], list):
                out_ids = [int(x) for x in payload["table_ids"]]
                if not out_ids and search_type in ("keyword", "single_column"):
                    print(
                        f"[tab2tab-debug] empty ids from Blend. "
                        f"search_type={search_type} query_arg_preview={query_arg_preview!r} "
                        f"tokens_preview={query_tokens[:20]!r}",
                        flush=True,
                    )
                return out_ids
            if "tableid" in payload and isinstance(payload["tableid"], list):
                out_ids = [int(x) for x in payload["tableid"]]
                if not out_ids and search_type in ("keyword", "single_column"):
                    print(
                        f"[tab2tab-debug] empty ids from Blend. "
                        f"search_type={search_type} query_arg_preview={query_arg_preview!r} "
                        f"tokens_preview={query_tokens[:20]!r}",
                        flush=True,
                    )
                return out_ids

        if isinstance(payload, list):
            # Case A: list of ints (or int-like strings)
            if all(isinstance(x, (int, float)) for x in payload):
                out_ids = [int(x) for x in payload]
                if not out_ids and search_type in ("keyword", "single_column"):
                    print(
                        f"[tab2tab-debug] empty ids from Blend. "
                        f"search_type={search_type} query_arg_preview={query_arg_preview!r} "
                        f"tokens_preview={query_tokens[:20]!r}",
                        flush=True,
                    )
                return out_ids
            if all(isinstance(x, str) and x.strip().isdigit() for x in payload):
                out_ids = [int(str(x).strip()) for x in payload]
                if not out_ids and search_type in ("keyword", "single_column"):
                    print(
                        f"[tab2tab-debug] empty ids from Blend. "
                        f"search_type={search_type} query_arg_preview={query_arg_preview!r} "
                        f"tokens_preview={query_tokens[:20]!r}",
                        flush=True,
                    )
                return out_ids

            # Case B: list of filenames / basenames -> map to tableid via DuckDB
            filenames = [os.path.basename(str(x).strip()) for x in payload if str(x).strip()]
            if not filenames:
                if search_type in ("keyword", "single_column"):
                    print(
                        f"[tab2tab-debug] Blend returned empty filenames list. "
                        f"search_type={search_type} query_arg_preview={query_arg_preview!r} "
                        f"tokens_preview={query_tokens[:20]!r}",
                        flush=True,
                    )
                return []

            import duckdb

            conn = duckdb.connect(db_path, read_only=True)
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
            if not out and search_type in ("keyword", "single_column"):
                print(
                    f"[tab2tab-debug] empty ids after filename->tableid mapping. "
                    f"search_type={search_type} query_arg_preview={query_arg_preview!r} "
                    f"tokens_preview={query_tokens[:20]!r}",
                    flush=True,
                )
            return out

        raise ValueError(f"Unexpected Blend_internal output JSON format: {payload!r}")
    finally:
        try:
            if not output_json and os.path.exists(out_path):
                os.remove(out_path)
        except Exception:
            pass

def main() -> None:
    """
    Minimal CLI for local testing (kept consistent with build_index.md usage).
    """
    import argparse
    import time
    parser = argparse.ArgumentParser(description="Table to Table Search using Blend_internal (subprocess wrapper)")
    parser.add_argument("--search_type", choices=["single_column", "multi_column", "keyword", "unionable"], required=True)
    parser.add_argument(
        "--query",
        default=None,
        required=True,
        help="Input table path/name. keyword extracts headers; single_column extracts first-column values; multi_column/unionable pass table path directly.",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output_json", default="", help="Optional output json path to write Blend_internal results.")
    parser.add_argument("--resources", nargs="+", default=["hugging"], choices=["hugging", "github", "arxiv"], help="Optional table resource filter on tab2tab results.")
    args = parser.parse_args()

    resources = [str(r).strip().lower() for r in (args.resources or []) if str(r).strip()]
    resource_set = set(resources)
    if resource_set == {'hugging'}:
        db_path = MODELLAKE_DB_HUGGING
    elif resource_set == {'hugging', 'github', 'arxiv'}:
        db_path = MODELLAKE_DB
    else:
        raise NotImplementedError(f"Unsupported resource combination: {resource_set}. Must be one of: {'hugging', 'github', 'arxiv'}")

    print(
        "[tab2tab CLI] artifacts: "
        f"resources={resources!r} | "
        f"modellake_db={os.path.abspath(db_path)} | "
        f"blend_repo={os.path.abspath(BLEND_INTERNAL_REPO or '')}",
        flush=True,
    )

    start = time.time()
    results = search_table2table(search_type=args.search_type, query=args.query, k=args.k, db_path=db_path, output_json=args.output_json)
    if args.output_json:
        print(f"Saved json: {args.output_json}")
    else:
        print(f"Found {len(results)} tables: {results}")
    print(f"Total time: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()

