"""
Check consistency between retrieved tables and integration inputs in local jobs.

Focus:
- Reconstruct retrieved table -> models from `search_results.json`
- Reconstruct integrated input table/model mappings from `integration_table_search*.json`
- Compare whether retrieve -> integrated pipeline keeps mappings correctly

Usage examples:

python scripts/check_retrieval_integration_consistency.py \
  --jobs-root data_251117/jobs_251117

python scripts/check_retrieval_integration_consistency.py \
  --jobs-root data_251117/jobs_251117 \
  --search-types single_column unionable keyword \
  --strict
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd


DEFAULT_SEARCH_TYPES = ["single_column", "unionable", "keyword"]
INTEGRATION_FILE_CANDIDATES = [
    "integration_table_search.json",
    "integration_table_search_alite_unionable_intermediate.json",
]


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON at {path} must be an object")
    return data


def _norm_mid(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, dict):
        mid = x.get("model_id") or x.get("modelId")
        if mid is None:
            return None
        s = str(mid).strip()
        return s or None
    s = str(x).strip()
    return s or None


def _unique_keep_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _canonical_table_to_models(mapping: Dict[str, Any]) -> Dict[str, Set[str]]:
    """
    Convert various table->models payloads to:
        {table_basename: {model_id, ...}}
    """
    out: Dict[str, Set[str]] = {}
    for table_key, raw_models in (mapping or {}).items():
        bn = os.path.basename(str(table_key))
        mids: List[str] = []
        if isinstance(raw_models, list):
            for m in raw_models:
                mid = _norm_mid(m)
                if mid:
                    mids.append(mid)
        out[bn] = set(_unique_keep_order(mids))
    return out


def _canonical_model_to_tables(mapping: Dict[str, Any]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for mid_key, raw_tables in (mapping or {}).items():
        mid = str(mid_key).strip()
        if not mid:
            continue
        tables: Set[str] = set()
        if isinstance(raw_tables, list):
            for t in raw_tables:
                bn = os.path.basename(str(t))
                if bn:
                    tables.add(bn)
        out[mid] = tables
    return out


def _invert_table_to_models(table_to_models: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    model_to_tables: Dict[str, Set[str]] = {}
    for tb, mids in table_to_models.items():
        for mid in mids:
            model_to_tables.setdefault(mid, set()).add(tb)
    return model_to_tables


def _extract_retrieved_from_search(search_results: Dict[str, Any], search_type: str) -> Tuple[Dict[str, Set[str]], str]:
    c2t2c = search_results.get("card2tab2card_results", {})
    block = c2t2c.get(search_type) if isinstance(c2t2c, dict) else None
    if not isinstance(block, dict):
        return {}, f"missing card2tab2card_results.{search_type}"

    intermediate = block.get("intermediate")
    if isinstance(intermediate, dict):
        table_to_models = _canonical_table_to_models(intermediate.get("table_to_models", {}))
        return table_to_models, "intermediate.table_to_models"

    mappings = block.get("mappings", {})
    if isinstance(mappings, dict):
        table_to_models = _canonical_table_to_models(mappings.get("retrieved_table_to_related_models", {}))
        if table_to_models:
            return table_to_models, "mappings.retrieved_table_to_related_models"

    return {}, f"no retrieved mapping for search_type={search_type}"


def _pick_integration_json(job_dir: str) -> Optional[str]:
    for name in INTEGRATION_FILE_CANDIDATES:
        p = os.path.join(job_dir, name)
        if os.path.isfile(p):
            return p
    return None


def _find_integration_jsons_by_search_type(job_dir: str) -> Dict[str, str]:
    """
    Return {search_type: json_path} for all integration jsons under job_dir.
    Falls back to file candidates if no wildcard matches.
    """
    out: Dict[str, str] = {}
    # best-effort glob without importing glob tool; keep it simple:
    for fn in sorted(os.listdir(job_dir)) if os.path.isdir(job_dir) else []:
        if not (fn.startswith("integration_table_search") and fn.endswith(".json")):
            continue
        p = os.path.join(job_dir, fn)
        if not os.path.isfile(p):
            continue
        try:
            obj = _load_json(p)
        except Exception:
            continue
        st = str(obj.get("search_type", "")).strip()
        if st:
            out[st] = p
    if out:
        return out
    # fallback: old fixed names (may only cover one type)
    p = _pick_integration_json(job_dir)
    if not p:
        return {}
    try:
        obj = _load_json(p)
    except Exception:
        return {}
    st = str(obj.get("search_type", "")).strip()
    if st:
        return {st: p}
    return {}


def _extract_retrieved_tables_from_search(search_results: Dict[str, Any], search_type: str) -> List[str]:
    """
    Return a list of retrieved table identifiers (paths or basenames) for a search_type.
    """
    c2t2c = search_results.get("card2tab2card_results", {})
    block = c2t2c.get(search_type) if isinstance(c2t2c, dict) else None
    if not isinstance(block, dict):
        return []
    if isinstance(block.get("searched_tables"), list) and block.get("searched_tables"):
        return [str(x) for x in block.get("searched_tables") if str(x).strip()]
    intermediate = block.get("intermediate")
    if isinstance(intermediate, dict) and isinstance(intermediate.get("retrieved_table_filenames"), list):
        return [str(x) for x in intermediate.get("retrieved_table_filenames") if str(x).strip()]
    mappings = block.get("mappings")
    if isinstance(mappings, dict) and isinstance(mappings.get("retrieved_table_to_related_models"), dict):
        return [str(k) for k in mappings["retrieved_table_to_related_models"].keys()]
    return []


@dataclass
class CheckResult:
    job_id: str
    search_type: str
    status: str  # OK | FAIL | SKIP
    notes: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "OK"


def _compare_one(job_dir: str, search_type: str) -> CheckResult:
    job_id = os.path.basename(os.path.normpath(job_dir))
    result = CheckResult(job_id=job_id, search_type=search_type, status="OK")

    search_path = os.path.join(job_dir, "search_results.json")
    if not os.path.isfile(search_path):
        result.status = "FAIL"
        result.errors.append("missing search_results.json")
        return result

    integration_path = _pick_integration_json(job_dir)
    if not integration_path:
        result.status = "SKIP"
        result.errors.append("missing integration_table_search*.json")
        return result

    try:
        search = _load_json(search_path)
        integ = _load_json(integration_path)
    except Exception as e:
        result.status = "FAIL"
        result.errors.append(f"json load error: {e}")
        return result

    expected_t2m, source = _extract_retrieved_from_search(search, search_type)
    if not expected_t2m:
        result.status = "FAIL"
        result.errors.append(f"cannot reconstruct retrieved table->models from {source}")
        return result

    integ_search_type = str(integ.get("search_type", "")).strip()
    if integ_search_type and integ_search_type != search_type:
        result.status = "SKIP"
        result.notes.append(
            f"integration search_type={integ_search_type}; skip expected={search_type}"
        )
        return result

    actual_t2m = _canonical_table_to_models(integ.get("model_to_table_paths", {}))  # wrong shape fallback
    # Build from integration's `retrieved_table_model_rows` first (closest to integration input)
    rows = integ.get("retrieved_table_model_rows", [])
    if isinstance(rows, list) and rows:
        tmp: Dict[str, Set[str]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            table = os.path.basename(str(row.get("table") or row.get("table_path") or "")).strip()
            if not table:
                continue
            mids: Set[str] = set()
            raw_models = row.get("models", [])
            if isinstance(raw_models, list):
                for m in raw_models:
                    mid = _norm_mid(m)
                    if mid:
                        mids.add(mid)
            tmp[table] = mids
        if tmp:
            actual_t2m = tmp
            result.notes.append("actual source: retrieved_table_model_rows")

    if not actual_t2m:
        # Fallback to model_to_table_paths inversion
        m2t = _canonical_model_to_tables(integ.get("model_to_table_paths", {}))
        actual_t2m = _invert_table_to_models({tb: set() for tb in []})  # empty init for type clarity
        actual_t2m = {}
        for mid, tables in m2t.items():
            for tb in tables:
                actual_t2m.setdefault(tb, set()).add(mid)
        result.notes.append("actual source: inverse(model_to_table_paths)")

    # Compare tables
    exp_tables = set(expected_t2m.keys())
    act_tables = set(actual_t2m.keys())
    missing_tables = sorted(exp_tables - act_tables)
    extra_tables = sorted(act_tables - exp_tables)
    if missing_tables:
        result.status = "FAIL"
        result.errors.append(f"missing tables in integrated inputs: {missing_tables[:10]}")
    if extra_tables:
        result.status = "FAIL"
        result.errors.append(f"unexpected extra tables in integrated inputs: {extra_tables[:10]}")

    # Compare model sets per table
    common_tables = sorted(exp_tables & act_tables)
    mismatch_cnt = 0
    mismatch_examples: List[str] = []
    for tb in common_tables:
        em = expected_t2m.get(tb, set())
        am = actual_t2m.get(tb, set())
        if em != am:
            mismatch_cnt += 1
            if len(mismatch_examples) < 3:
                miss = sorted(em - am)
                extra = sorted(am - em)
                mismatch_examples.append(
                    f"{tb}: missing_models={miss[:5]}, extra_models={extra[:5]}"
                )
    if mismatch_cnt > 0:
        result.status = "FAIL"
        result.errors.append(f"table->models mismatch count={mismatch_cnt}; examples={mismatch_examples}")

    result.notes.append(f"retrieved mapping source: {source}")
    result.notes.append(f"integration json: {os.path.basename(integration_path)}")
    result.notes.append(f"expected tables={len(exp_tables)}, actual tables={len(act_tables)}")
    return result


def _iter_job_dirs(jobs_root: str) -> List[str]:
    if not os.path.isdir(jobs_root):
        return []
    out: List[str] = []
    for name in sorted(os.listdir(jobs_root)):
        p = os.path.join(jobs_root, name)
        if os.path.isdir(p) and name not in {"batch_runs", "job_md"}:
            out.append(p)
    return out


def _render_markdown_report(
    all_results: List[CheckResult],
    *,
    jobs_root: str,
    search_types: List[str],
) -> str:
    total = len(all_results)
    fail = sum(1 for r in all_results if r.status == "FAIL")
    ok = sum(1 for r in all_results if r.status == "OK")
    skip = sum(1 for r in all_results if r.status == "SKIP")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: List[str] = []
    lines.append("# Retrieval vs Integration Consistency Report")
    lines.append("")
    lines.append(f"- Generated at: `{ts}`")
    lines.append(f"- Jobs root: `{jobs_root}`")
    lines.append(f"- Search types: `{', '.join(search_types)}`")
    lines.append(f"- Summary: total={total}, ok={ok}, fail={fail}, skip={skip}")
    lines.append("")
    lines.append("## Failed Checks")
    lines.append("")

    failed = [r for r in all_results if r.status == "FAIL"]
    if not failed:
        lines.append("No failed checks.")
    else:
        for r in failed:
            lines.append(f"### `{r.job_id}` / `{r.search_type}`")
            lines.append("")
            for e in r.errors:
                lines.append(f"- error: {e}")
            for n in r.notes:
                lines.append(f"- note: {n}")
            lines.append("")

    lines.append("## All Checks")
    lines.append("")
    lines.append("| Job ID | Search Type | Status |")
    lines.append("|---|---|---|")
    for r in all_results:
        lines.append(f"| `{r.job_id}` | `{r.search_type}` | `{r.status}` |")
    lines.append("")
    return "\n".join(lines)


def _md_escape(x: Any) -> str:
    s = str(x)
    s = s.replace("\n", " ").replace("\r", " ")
    return s.replace("|", "\\|")


def _df_to_markdown(
    df: pd.DataFrame,
    max_rows: Optional[int] = None,
    max_cols: Optional[int] = None,
    max_cell: int = 60,
) -> str:
    if df is None or df.empty:
        return "_(empty table)_"
    row_end = max_rows if (isinstance(max_rows, int) and max_rows > 0) else len(df)
    col_end = max_cols if (isinstance(max_cols, int) and max_cols > 0) else len(df.columns)
    sub = df.iloc[:row_end, :col_end].copy()
    cols = [str(c) for c in sub.columns]
    lines: List[str] = []
    lines.append("| " + " | ".join(_md_escape(c) for c in cols) + " |")
    lines.append("|" + "|".join("---" for _ in cols) + "|")
    for _, row in sub.iterrows():
        vals: List[str] = []
        for v in row.tolist():
            t = _md_escape(v)
            if len(t) > max_cell:
                t = t[: max_cell - 3] + "..."
            vals.append(t)
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _extract_integrated_df(integ: Dict[str, Any]) -> pd.DataFrame:
    payload = integ.get("integrated_table", {})
    cols = payload.get("columns", []) if isinstance(payload, dict) else []
    data = payload.get("data", []) if isinstance(payload, dict) else []
    if not isinstance(cols, list) or not isinstance(data, list):
        return pd.DataFrame()
    try:
        return pd.DataFrame(data, columns=cols)
    except Exception:
        return pd.DataFrame()


def _resolve_query_table_df(query_table_path_or_name: str) -> pd.DataFrame:
    """
    Resolve and load query table CSV robustly without importing `src.utils`.

    Resolution order:
    1) Use path as-is if it exists
    2) If path starts with `/u1/...`, remap to local `/Users/...`
    3) Try basename under `TABLE_BASE_DIRS` from config
    """
    try:
        from src.config import TABLE_BASE_DIRS
    except Exception:
        TABLE_BASE_DIRS = []

    raw = str(query_table_path_or_name or "").strip()
    if not raw:
        return pd.DataFrame()

    candidates: List[str] = []
    candidates.append(raw)

    if raw.startswith("/u1/"):
        # Cluster path style: /u1/<user>/Repo/...
        # Local mac style:     /Users/<local_user>/Repo/...
        local_user = os.path.basename(os.path.expanduser("~"))
        m = re.match(r"^/u1/[^/]+/(.*)$", raw)
        if m:
            candidates.append("/Users/" + local_user + "/" + m.group(1))
        candidates.append("/Users/" + raw[len("/u1/") :])

    bn = os.path.basename(raw)
    if bn:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        repo_parent = os.path.dirname(repo_root)
        explicit_dirs = [
            os.path.join(repo_parent, "ModelTables", "data", "processed", "deduped_hugging_csvs_v2_251117"),
            os.path.join(repo_parent, "ModelTables", "data", "processed", "deduped_github_csvs_v2_251117"),
            os.path.join(repo_parent, "ModelTables", "data", "processed", "tables_output_v2_251117"),
        ]
        for d in explicit_dirs:
            candidates.append(os.path.join(d, bn))
        for d in TABLE_BASE_DIRS:
            candidates.append(os.path.join(str(d), bn))

    seen: Set[str] = set()
    for p in candidates:
        ap = os.path.abspath(p)
        if ap in seen:
            continue
        seen.add(ap)
        if not os.path.isfile(ap):
            continue
        try:
            return pd.read_csv(ap)
        except Exception:
            continue
    return pd.DataFrame()


def _render_preview_markdown(
    job_dirs: List[str],
    *,
    preview_search_type: str,
    job_limit: int,
    preview_retrieved_table_limit: int,
    preview_max_rows: Optional[int],
    preview_max_cols: Optional[int],
) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = []
    lines.append("# Retrieved Inputs vs Integrated Table Preview")
    lines.append("")
    lines.append(f"- Generated at: `{ts}`")
    lines.append(f"- Search type: `{preview_search_type}`")
    lines.append(f"- Job count: `{min(len(job_dirs), job_limit)}`")
    rt_limit_label = "all" if preview_retrieved_table_limit <= 0 else str(preview_retrieved_table_limit)
    lines.append(f"- Retrieved table preview per job: `{rt_limit_label}`")
    lines.append("")

    shown = 0
    for job_dir in job_dirs:
        if shown >= job_limit:
            break
        search_path = os.path.join(job_dir, "search_results.json")
        integ_path = _pick_integration_json(job_dir)
        if not os.path.isfile(search_path) or not integ_path:
            continue
        try:
            search = _load_json(search_path)
            integ = _load_json(integ_path)
        except Exception:
            continue

        integ_st = str(integ.get("search_type", "")).strip()
        if integ_st and integ_st != preview_search_type:
            continue

        block = (search.get("card2tab2card_results", {}) or {}).get(preview_search_type, {})
        integrated_df = _extract_integrated_df(integ)
        query_tables = block.get("query_tables", []) if isinstance(block, dict) else []
        qtb = query_tables[0] if isinstance(query_tables, list) and query_tables else ""
        query_df = _resolve_query_table_df(str(qtb)) if qtb else pd.DataFrame()

        raw_table_paths = integ.get("table_paths", [])
        retrieved_paths: List[str] = []
        if isinstance(raw_table_paths, list):
            retrieved_paths = [str(p).strip() for p in raw_table_paths if str(p).strip()]

        job_id = os.path.basename(os.path.normpath(job_dir))

        lines.append(f"## `{job_id}`")
        lines.append("")
        lines.append(f"- query_table (seed only): `{qtb or '(missing)'}`")
        lines.append(f"- retrieved tables used for integration: `{len(retrieved_paths)}`")
        lines.append(
            f"- integrated_table shape: `{integrated_df.shape[0]} x {integrated_df.shape[1]}`"
        )
        lines.append(f"- integration_file: `{os.path.basename(integ_path)}`")
        lines.append("")
        lines.append("### Retrieved Table Paths (integration inputs)")
        lines.append("")
        if retrieved_paths:
            for p in retrieved_paths:
                lines.append(f"- `{p}`")
        else:
            lines.append("_(no table_paths in integration json)_")
        lines.append("")
        lines.append("### Retrieved Tables (head previews)")
        lines.append("")
        if retrieved_paths:
            table_iter = retrieved_paths if preview_retrieved_table_limit <= 0 else retrieved_paths[:preview_retrieved_table_limit]
            for i, p in enumerate(table_iter, start=1):
                rdf = _resolve_query_table_df(p)
                lines.append(f"#### Retrieved table {i}: `{os.path.basename(p)}`")
                lines.append("")
                lines.append(_df_to_markdown(rdf, max_rows=preview_max_rows, max_cols=preview_max_cols))
                lines.append("")
        else:
            lines.append("_(empty)_")
            lines.append("")

        lines.append("### Query Table (head, reference only)")
        lines.append("")
        lines.append(_df_to_markdown(query_df, max_rows=preview_max_rows, max_cols=preview_max_cols))
        lines.append("")
        lines.append("### Integrated Table (head)")
        lines.append("")
        lines.append(_df_to_markdown(integrated_df, max_rows=preview_max_rows, max_cols=preview_max_cols))
        lines.append("")
        shown += 1

    if shown == 0:
        lines.append("_No preview generated (no matching jobs/files)._")
        lines.append("")
    return "\n".join(lines)


def _slug_anchor(s: str) -> str:
    s2 = re.sub(r"[^a-zA-Z0-9_ -]+", "", str(s)).strip().lower()
    return s2.replace(" ", "-")


def _write_job_md(
    *,
    job_dir: str,
    out_path: str,
    search_types: List[str],
    preview_max_rows: Optional[int],
    preview_max_cols: Optional[int],
) -> None:
    search_path = os.path.join(job_dir, "search_results.json")
    if not os.path.isfile(search_path):
        return
    try:
        search = _load_json(search_path)
    except Exception:
        return

    integ_map = _find_integration_jsons_by_search_type(job_dir)
    job_id = os.path.basename(os.path.normpath(job_dir))
    query_text = str(search.get("query", "")).strip()

    lines: List[str] = []
    lines.append(f"# Job `{job_id}`")
    lines.append("")
    lines.append(f"- job_dir: `{job_dir}`")
    lines.append(f"- search_results: `{search_path}`")
    lines.append("")
    lines.append("## Query (NLP)")
    lines.append("")
    lines.append(f"`{query_text}`" if query_text else "_(missing)_")
    lines.append("")

    # global query table (seed) per search_type may differ; show per section
    lines.append("## Directory")
    lines.append("")
    for st in search_types:
        anchor = _slug_anchor(st)
        lines.append(f"- [{st}](#{anchor})")
    lines.append("")

    c2t2c = search.get("card2tab2card_results", {}) if isinstance(search, dict) else {}

    for st in search_types:
        anchor = _slug_anchor(st)
        lines.append(f"## {st}")
        lines.append("")

        block = c2t2c.get(st) if isinstance(c2t2c, dict) else None
        if not isinstance(block, dict):
            lines.append("_(no card2tab2card results for this search type)_")
            lines.append("")
            continue

        query_tables = block.get("query_tables", []) if isinstance(block.get("query_tables"), list) else []
        qtb = str(query_tables[0]).strip() if query_tables else ""
        lines.append(f"- query_table: `{qtb or '(missing)'}`")

        retrieved = _extract_retrieved_tables_from_search(search, st)
        lines.append(f"- retrieved_tables: `{len(retrieved)}`")

        integ_path = integ_map.get(st)
        if integ_path and os.path.isfile(integ_path):
            try:
                integ = _load_json(integ_path)
            except Exception:
                integ = {}
            integrated_df = _extract_integrated_df(integ) if isinstance(integ, dict) else pd.DataFrame()
            table_paths = integ.get("table_paths", []) if isinstance(integ, dict) else []
            table_paths = [str(p).strip() for p in table_paths if str(p).strip()] if isinstance(table_paths, list) else []
            lines.append(f"- integration_json: `{os.path.basename(integ_path)}`")
            lines.append(f"- integrated_table shape: `{integrated_df.shape[0]} x {integrated_df.shape[1]}`")
        else:
            integ = {}
            integrated_df = pd.DataFrame()
            table_paths = []
            lines.append("- integration_json: `None`")
            lines.append("- integrated_table shape: `0 x 0`")
        lines.append("")

        lines.append("### Query table")
        lines.append("")
        qdf = _resolve_query_table_df(qtb) if qtb else pd.DataFrame()
        lines.append(_df_to_markdown(qdf, max_rows=preview_max_rows, max_cols=preview_max_cols))
        lines.append("")

        lines.append("### Retrieved tables")
        lines.append("")
        # Prefer showing retrieved tables from search payload; if empty, fall back to integration inputs (table_paths).
        retrieved_for_preview = retrieved if retrieved else table_paths
        if retrieved_for_preview:
            for i, p in enumerate(retrieved_for_preview, start=1):
                bn = os.path.basename(str(p))
                lines.append(f"#### {i}. `{bn}`")
                lines.append("")
                tdf = _resolve_query_table_df(str(p))
                lines.append(_df_to_markdown(tdf, max_rows=preview_max_rows, max_cols=preview_max_cols))
                lines.append("")
        else:
            lines.append("_(empty)_")
            lines.append("")

        lines.append("### Integrated table")
        lines.append("")
        lines.append(_df_to_markdown(integrated_df, max_rows=preview_max_rows, max_cols=preview_max_cols))
        lines.append("")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check retrieve->integrated mapping consistency for local jobs"
    )
    parser.add_argument(
        "--jobs-root",
        type=str,
        default="data_251117/jobs_251117",
        help="Root directory that contains multiple job folders",
    )
    parser.add_argument(
        "--job-dir",
        type=str,
        default=None,
        help="Single job directory to check (optional)",
    )
    parser.add_argument(
        "--search-types",
        nargs="+",
        default=DEFAULT_SEARCH_TYPES,
        help="Search types to validate, e.g. single_column unionable keyword",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero exit code when any check fails",
    )
    parser.add_argument(
        "--markdown-out",
        type=str,
        default=None,
        help=(
            "Write Markdown report to this path. "
            "Default: <jobs-root>/retrieval_integration_consistency_report.md"
        ),
    )
    parser.add_argument(
        "--no-summary-md",
        action="store_true",
        help="Do not write the aggregate consistency report markdown.",
    )
    parser.add_argument(
        "--preview-md-out",
        type=str,
        default=None,
        help=(
            "Write preview markdown with query/integrated table heads. "
            "Default: <jobs-root>/query_integrated_preview.md"
        ),
    )
    parser.add_argument(
        "--no-preview-md",
        action="store_true",
        help="Do not write the aggregate preview markdown.",
    )
    parser.add_argument(
        "--preview-search-type",
        type=str,
        default="unionable",
        help="Which search_type to preview in table snapshots (default: unionable)",
    )
    parser.add_argument(
        "--preview-job-limit",
        type=int,
        default=20,
        help="Max number of jobs included in preview markdown",
    )
    parser.add_argument(
        "--preview-retrieved-table-limit",
        type=int,
        default=0,
        help="How many retrieved input tables to show per job; <=0 means all",
    )
    parser.add_argument(
        "--preview-max-rows",
        type=int,
        default=0,
        help="Max rows per rendered table in preview markdown; <=0 means all rows",
    )
    parser.add_argument(
        "--preview-max-cols",
        type=int,
        default=0,
        help="Max cols per rendered table in preview markdown; <=0 means all cols",
    )
    parser.add_argument(
        "--per-job-md",
        action="store_true",
        help="Generate one markdown file per job (recommended).",
    )
    parser.add_argument(
        "--per-job-md-dir",
        type=str,
        default=None,
        help="Output directory for per-job markdown files (default: <jobs-root>/job_md)",
    )

    args = parser.parse_args(argv)
    search_types = [str(x).strip() for x in args.search_types if str(x).strip()]

    if args.job_dir:
        job_dirs = [args.job_dir]
    else:
        job_dirs = _iter_job_dirs(args.jobs_root)

    if not job_dirs:
        print(f"❌ no job dirs found. jobs_root={args.jobs_root}", file=sys.stderr)
        return 2

    all_results: List[CheckResult] = []
    for job_dir in job_dirs:
        for st in search_types:
            r = _compare_one(job_dir, st)
            all_results.append(r)
            status = r.status
            print(f"[{status}] {r.job_id} | search_type={r.search_type}")
            for e in r.errors:
                print(f"  - error: {e}")
            for n in r.notes:
                print(f"  - note: {n}")

    total = len(all_results)
    fail = sum(1 for r in all_results if r.status == "FAIL")
    ok = sum(1 for r in all_results if r.status == "OK")
    skip = sum(1 for r in all_results if r.status == "SKIP")
    print("\n" + "=" * 60)
    print(f"Summary: total={total}, ok={ok}, fail={fail}, skip={skip}")
    print("=" * 60)

    # By default, when generating per-job markdown, skip aggregate md outputs unless explicitly requested.
    write_summary_md = not args.no_summary_md and (not args.per_job_md)
    write_preview_md = not args.no_preview_md and (not args.per_job_md)

    if write_summary_md:
        md_out = args.markdown_out or os.path.join(
            args.jobs_root, "retrieval_integration_consistency_report.md"
        )
        md = _render_markdown_report(
            all_results,
            jobs_root=args.jobs_root,
            search_types=search_types,
        )
        os.makedirs(os.path.dirname(os.path.abspath(md_out)), exist_ok=True)
        with open(md_out, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"📄 Markdown report written: {md_out}")

    if write_preview_md:
        preview_out = args.preview_md_out or os.path.join(args.jobs_root, "query_integrated_preview.md")
        preview_md = _render_preview_markdown(
            job_dirs,
            preview_search_type=str(args.preview_search_type).strip(),
            job_limit=max(1, int(args.preview_job_limit)),
            preview_retrieved_table_limit=int(args.preview_retrieved_table_limit),
            preview_max_rows=(None if int(args.preview_max_rows) <= 0 else int(args.preview_max_rows)),
            preview_max_cols=(None if int(args.preview_max_cols) <= 0 else int(args.preview_max_cols)),
        )
        os.makedirs(os.path.dirname(os.path.abspath(preview_out)), exist_ok=True)
        with open(preview_out, "w", encoding="utf-8") as f:
            f.write(preview_md)
        print(f"📄 Preview markdown written: {preview_out}")

    if args.per_job_md:
        out_dir = args.per_job_md_dir or os.path.join(args.jobs_root, "job_md")
        os.makedirs(out_dir, exist_ok=True)
        index_lines: List[str] = []
        index_lines.append("# Job Markdown Index")
        index_lines.append("")
        index_lines.append(f"- jobs_root: `{args.jobs_root}`")
        index_lines.append(f"- output_dir: `{out_dir}`")
        index_lines.append("")
        index_lines.append("## Jobs")
        index_lines.append("")

        for jd in job_dirs:
            jid = os.path.basename(os.path.normpath(jd))
            out_path = os.path.join(out_dir, f"{jid}.md")
            _write_job_md(
                job_dir=jd,
                out_path=out_path,
                search_types=search_types,
                preview_max_rows=(None if int(args.preview_max_rows) <= 0 else int(args.preview_max_rows)),
                preview_max_cols=(None if int(args.preview_max_cols) <= 0 else int(args.preview_max_cols)),
            )
            rel = os.path.relpath(out_path, start=out_dir)
            index_lines.append(f"- [`{jid}`]({rel})")

        index_path = os.path.join(out_dir, "README.md")
        with open(index_path, "w", encoding="utf-8") as f:
            f.write("\n".join(index_lines))
        print(f"📄 Per-job markdown index written: {index_path}")

    if args.strict and fail > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
