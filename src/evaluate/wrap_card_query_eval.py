#!/usr/bin/env python3
"""One job from batch JSON: card2nugget (per-model CSVs under data_*/card2nugget/, existing files reused)
-> query2nugget -> per-method qrels/run -> eval. Invoke once per --job-id."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Literal

from src.config import OUTPUT_DIR
from src.evaluate.card2nugget_extraction import CARD2NUGGET_DIR, run_batch, _safe_model_id
from src.evaluate.evaluate_pyndeval import load_run, load_subtopic_qrels, mean
from src.evaluate.query2nugget_mapping import (
    NUGGET_SCHEMA_HEADERS,
    build_qrels_and_run_llm_rerank,
    build_qrels_and_run_structured,
    count_csv_data_rows,
    map_queries,
    _header_non_empty_for_row,
    _row_dict,
)

EVAL_DIR = Path(OUTPUT_DIR) / "evaluate"
PIPELINE_DIR = EVAL_DIR / "pipeline"


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _safe_name(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", (text or "").strip())
    return s.strip("_") or "item"


def _write_qrels(path: Path, rows: list[tuple[str, str, str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid, subtopic, doc_id, rel in rows:
            f.write(f"{qid} {subtopic} {doc_id} {rel}\n")


def _write_run(path: Path, rows: list[tuple[str, str, str, int, float, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid, q0, doc_id, rank, score, tag in rows:
            f.write(f"{qid} {q0} {doc_id} {rank} {score} {tag}\n")


def _extract_job_sets(batch_json: Path, wanted_job_ids: set[str] | None) -> list[dict[str, Any]]:
    payload = json.loads(batch_json.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    jobs_root = batch_json.parent.parent
    sets: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        job_id = str(item.get("job_id", "")).strip()
        if wanted_job_ids and job_id not in wanted_job_ids:
            continue
        query = str(item.get("query", "")).strip()
        method_model_sets = _extract_method_model_sets(item, jobs_root)
        model_ids = _unique_keep(mid for ms in method_model_sets for mid in ms.get("model_ids", []))
        if not query or not model_ids:
            continue
        sets.append({"job_id": job_id, "query": query, "model_ids": model_ids, "method_model_sets": method_model_sets})
    return sets


def _unique_keep(items: list[str] | tuple[str, ...] | Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for x in items:
        s = str(x).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _job_artifact_dir(item: dict[str, Any], jobs_root: Path) -> Path:
    search_resp = item.get("search_response")
    if isinstance(search_resp, dict):
        folder_path = str(search_resp.get("folder_path", "")).strip()
        if folder_path:
            return jobs_root / Path(folder_path).name
    return jobs_root / str(item.get("job_id", "")).strip()


def _first_n(items: list[str], n: int) -> list[str]:
    return items[: max(0, n)] if n > 0 else items


def _extract_method_model_sets(item: dict[str, Any], jobs_root: Path) -> list[dict[str, Any]]:
    job_dir = _job_artifact_dir(item, jobs_root)
    search_resp = item.get("search_response") if isinstance(item.get("search_response"), dict) else {}
    model_top_k = int(search_resp.get("model_top_k", 3) or 3)
    ims = item.get("integration_model_search")
    out_by_method: dict[str, dict[str, Any]] = {}

    q2mc = _load_json_if_exists(job_dir / "query2modelcard.json")
    q2mc_results = q2mc.get("results", {}) if isinstance(q2mc.get("results"), dict) else {}
    if isinstance(ims, dict):
        for method, payload in ims.items():
            if not isinstance(payload, dict):
                continue
            ids_raw = payload.get("model_ids") or payload.get("models_with_tables") or []
            model_ids = _unique_keep(ids_raw if isinstance(ids_raw, list) else [])
            if not model_ids and isinstance(q2mc_results.get(method), list):
                model_ids = _first_n(_unique_keep(q2mc_results.get(method, [])), model_top_k)
            status = str(payload.get("status", "")).strip() or ("success" if model_ids else "")
            note = str(payload.get("message", "")).strip()
            meth = str(method).strip() or "unknown"
            if model_ids or status or note:
                out_by_method[meth] = {"method": meth, "model_ids": model_ids, "status": status, "note": note}

    for method in ("dense", "sparse", "hybrid"):
        if method in out_by_method:
            continue
        if isinstance(q2mc_results.get(method), list):
            mids = _first_n(_unique_keep(q2mc_results.get(method, [])), model_top_k)
            out_by_method[method] = {"method": method, "model_ids": mids, "status": "success" if mids else "", "note": ""}

    for method in ("keyword", "single_column", "unionable"):
        c2t2c = _load_json_if_exists(job_dir / f"card2tab2card_{method}.json")
        model_ids: list[str] = []
        if isinstance(c2t2c.get("model_rerank_map"), list):
            model_ids = _first_n(_unique_keep(c2t2c.get("model_rerank_map", [])), model_top_k)
        if not model_ids and isinstance(c2t2c.get("tab2card_map"), dict):
            flat: list[str] = []
            for vals in c2t2c["tab2card_map"].values():
                if isinstance(vals, list):
                    flat.extend(vals)
            model_ids = _first_n(_unique_keep(flat), model_top_k)
        if model_ids:
            out_by_method[method] = {"method": method, "model_ids": model_ids, "status": "success", "note": ""}

    order = ["sparse", "dense", "hybrid", "keyword", "single_column", "unionable"]
    out = [out_by_method[m] for m in order if m in out_by_method]
    for m in sorted(k for k in out_by_method if k not in order):
        out.append(out_by_method[m])
    return out


def _split_existing_batch_models(model_ids: list[str]) -> tuple[list[str], list[dict[str, str]], list[str]]:
    to_run: list[str] = []
    reused: list[dict[str, str]] = []
    skipped_ids: list[str] = []
    for model_id in model_ids:
        csv_path = CARD2NUGGET_DIR / f"{_safe_model_id(model_id)}.csv"
        meta_path = CARD2NUGGET_DIR / f"{_safe_model_id(model_id)}_meta.yaml"
        if csv_path.is_file():
            reused.append({"model_id": model_id, "csv_path": str(csv_path.resolve()), "meta_path": str(meta_path.resolve()), "note": "exists_skip"})
            skipped_ids.append(model_id)
        else:
            to_run.append(model_id)
    return to_run, reused, skipped_ids


def _headers_from_query_maps(query_maps: list[dict[str, Any]]) -> list[str]:
    if not query_maps:
        return []
    first = query_maps[0]
    hl = first.get("header_list", [])
    if isinstance(hl, list):
        out = _unique_keep(hl)
        if out:
            return out
    clusters = first.get("clusters", [])
    if not isinstance(clusters, list):
        return []
    headers: list[str] = []
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        related = cluster.get("related", [])
        if not isinstance(related, list):
            continue
        for item in related:
            if isinstance(item, dict):
                h = str(item.get("header", "")).strip()
                if h:
                    headers.append(h)
    return _unique_keep(headers)


def _count_rows_with_any_header(csv_path: Path, headers: list[str]) -> int:
    if not headers or not csv_path.is_file():
        return 0
    total = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cells = _row_dict(row)
            if any(_header_non_empty_for_row(h, cells) for h in headers):
                total += 1
    return total


def _nonempty_headers_for_csv(csv_path: Path) -> list[str]:
    if not csv_path.is_file():
        return []
    seen: list[str] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cells = _row_dict(row)
            for header, value in cells.items():
                if header == "source_model_id":
                    continue
                if value and header not in seen:
                    seen.append(header)
    return seen


def _row_signature(cells: dict[str, str]) -> tuple[str, ...]:
    return tuple((cells.get(h, "") or "").strip() for h in NUGGET_SCHEMA_HEADERS)


def _write_method_dedup_csv(path: Path, method_csv_paths: list[Path], model_ids: list[str]) -> int:
    """Merge one method's model CSVs into a deduplicated nugget CSV for inspection."""
    by_mid = {str(mid).strip(): Path(p) for mid, p in zip(model_ids, method_csv_paths)}
    dedup: dict[tuple[str, ...], dict[str, str]] = {}
    provenance: dict[tuple[str, ...], list[str]] = {}
    for mid in model_ids:
        csv_path = by_mid.get(str(mid).strip())
        if not csv_path or not csv_path.is_file():
            continue
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cells = _row_dict(row)
                sig = _row_signature(cells)
                if sig not in dedup:
                    dedup[sig] = {h: cells.get(h, "") for h in NUGGET_SCHEMA_HEADERS}
                    provenance[sig] = []
                if mid not in provenance[sig]:
                    provenance[sig].append(mid)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["source_model_ids"] + list(NUGGET_SCHEMA_HEADERS)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for sig, row in dedup.items():
            out_row = {"source_model_ids": " | ".join(provenance.get(sig, []))}
            out_row.update(row)
            w.writerow(out_row)
    return len(dedup)


def _collect_method_row_stats(csv_paths: list[Path], headers: list[str]) -> dict[str, Any]:
    raw_count = 0
    matched_count = 0
    raw_signatures: set[tuple[str, ...]] = set()
    matched_signatures: set[tuple[str, ...]] = set()
    for csv_path in csv_paths:
        if not csv_path.is_file():
            continue
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cells = _row_dict(row)
                sig = _row_signature(cells)
                raw_count += 1
                raw_signatures.add(sig)
                if headers and any(_header_non_empty_for_row(h, cells) for h in headers):
                    matched_count += 1
                    matched_signatures.add(sig)
    return {
        "raw_count": raw_count,
        "matched_count": matched_count,
        "raw_dedup_count": len(raw_signatures),
        "matched_dedup_count": len(matched_signatures),
        "raw_signatures": raw_signatures,
        "matched_signatures": matched_signatures,
    }


def _md_file_link(path_str: str) -> str:
    path = str(path_str or "").strip()
    if not path:
        return "—"
    name = Path(path).name.replace("|", "\\|")
    return f"[{name}](file://{path})"


def _format_pipeline_match_markdown(
    *,
    jobs_path: Path,
    job_id: str,
    query: str,
    card_rows: list[dict[str, Any]],
    query_headers: list[str],
    query_method_counts: list[dict[str, Any]],
) -> str:
    lines = [
        "# Pipeline summary",
        "",
        f"- Jobs JSON: `{jobs_path.resolve()}`",
        f"- job_id: `{job_id}`",
        "",
        "## Card2nugget",
        "",
        "| method | model_id | nugget_rows | csv_path | nonempty_headers |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for row in card_rows:
        csv_link = _md_file_link(str(row.get("csv_path", "")))
        lines.append(
            f"| `{row['method']}` | `{row['model_id']}` | {row['nugget_rows']} | "
            f"{csv_link} | "
            f"{str(row.get('nonempty_headers', '')).replace('|', '\\|')} |"
        )
    if not card_rows:
        lines.append("| — | — | 0 | — | — |")

    lines.extend(["", "## Query2nugget", ""])
    lines.append(f"- Query: `{_md_query_cell(query)}`")
    lines.append(f"- Headers: `{', '.join(query_headers) if query_headers else '[]'}`")
    lines.append("")
    if query_method_counts:
        lines.extend(
            [
                "| method | model_id | original | filter |",
                "| --- | --- | ---: | ---: |",
            ]
        )
        for row in query_method_counts:
            method = str(row.get("method", ""))
            for model in row.get("models", []):
                lines.append(
                    f"| `{method}` | `{model.get('model_id', '')}` | "
                    f"{int(model.get('raw_rows', 0))} | {int(model.get('matched_rows', 0))} |"
                )
            lines.append(
                f"| `{method}` | `sum` | {int(row.get('raw_rows', 0))} | {int(row.get('matched_rows', 0))} |"
            )
            lines.append(
                f"| `{method}` | `dedup` | {int(row.get('raw_dedup_count', 0))} | {int(row.get('matched_dedup_count', 0))} |"
            )
    else:
        lines.extend([
            "| method | model_id | original | filter |",
            "| --- | --- | ---: | ---: |",
            "| — | — | 0 | 0 |",
        ])
    lines.extend(
        [
            "",
            "_`original` = raw nugget rows from card2nugget before header filtering; `filter` = rows where any query-selected header is non-empty. `dedup` is the unique nugget count within that method after deduplicating across its model cards._",
            "",
            "## Method Summary",
            "",
            "| method | model_count | original_sum | original_dedup | filter_sum | filter_dedup | nugget_csv |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in query_method_counts:
        lines.append(
            f"| `{row.get('method', '')}` | {len(row.get('models', []))} | "
            f"{int(row.get('raw_rows', 0))} | {int(row.get('raw_dedup_count', 0))} | "
            f"{int(row.get('matched_rows', 0))} | {int(row.get('matched_dedup_count', 0))} | "
            f"{_md_file_link(str(row.get('nugget_csv_path', '')))} |"
        )
    lines.extend(
        [
            "",
            "_`sum` = direct sum over model cards in that method; `dedup` = unique nugget count within that method after deduplication._",
            "",
            "_No alpha-nDCG / strec here: this setting is open-world and we are comparing matched nugget counts, not coverage against a fixed ground-truth set._",
        ]
    )
    return "\n".join(lines) + "\n"


def _md_query_cell(s: str) -> str:
    return (s or "").replace("|", "\\|").replace("\n", " ")


def _run_cluster(
    cluster_name: str,
    job_id: str,
    model_ids: list[str],
    method_model_sets: list[dict[str, Any]],
    query_maps: list[dict[str, Any]],
    out_dir: Path,
    subtopic: str,
    *,
    llm_mode: Literal["batch", "iter"],
    match_build: Literal["structured", "llm_rerank"] = "structured",
    rerank_model: str | None = None,
) -> dict[str, Any]:
    print(f"[cluster:{cluster_name}] card2nugget candidates: {len(model_ids)} model(s)")
    to_run, reused_outputs, skipped_ids = _split_existing_batch_models(model_ids)
    print(f"[cluster:{cluster_name}] saved_model_ids={len(skipped_ids)} to_create={len(to_run)} total={len(model_ids)}")
    if skipped_ids:
        print(f"[cluster:{cluster_name}] skipped existing model ids: {skipped_ids}")

    new_outputs = run_batch(to_run, llm_mode=llm_mode) if to_run else []
    card_outputs = reused_outputs + new_outputs
    per_model_csv_paths = [Path(x["csv_path"]) for x in card_outputs if x.get("csv_path")]
    by_model_id = {str(x.get("model_id", "")).strip(): x for x in card_outputs if str(x.get("model_id", "")).strip()}
    card_rows: list[dict[str, Any]] = []
    query_headers = _headers_from_query_maps(query_maps)
    query_method_counts: list[dict[str, Any]] = []
    method_runs: list[dict[str, Any]] = []
    qrels_builder = build_qrels_and_run_llm_rerank if match_build == "llm_rerank" else build_qrels_and_run_structured
    for method_info in method_model_sets:
        method = str(method_info.get("method", "")).strip() or "unknown"
        method_model_ids = _unique_keep(method_info.get("model_ids", []))
        method_csv_paths: list[Path] = []
        for mid in method_model_ids:
            co = by_model_id.get(mid)
            csv_path = Path(str(co.get("csv_path", ""))) if co else Path()
            nugget_rows = count_csv_data_rows(csv_path) if csv_path.is_file() else 0
            filtered_rows = _count_rows_with_any_header(csv_path, query_headers) if csv_path.is_file() else 0
            nonempty_headers = ", ".join(_nonempty_headers_for_csv(csv_path))
            card_rows.append(
                {
                    "method": method,
                    "model_id": mid,
                    "nugget_rows": nugget_rows,
                    "filtered_rows": filtered_rows,
                    "csv_path": str(csv_path.resolve()) if csv_path.is_file() else "",
                    "nonempty_headers": nonempty_headers,
                }
            )
            if csv_path.is_file():
                method_csv_paths.append(csv_path)
        stats = _collect_method_row_stats(method_csv_paths, query_headers)
        model_rows = [r for r in card_rows if r["method"] == method]
        query_method_counts.append(
            {
                "method": method,
                "models": [
                    {"model_id": row["model_id"], "raw_rows": row["nugget_rows"], "matched_rows": row["filtered_rows"]}
                    for row in model_rows
                ],
                "raw_rows": stats["raw_count"],
                "matched_rows": stats["matched_count"],
                "raw_dedup_count": stats["raw_dedup_count"],
                "matched_dedup_count": stats["matched_dedup_count"],
                "nugget_csv_path": "",
            }
        )
        if not method_csv_paths:
            method_runs.append(
                {
                    "method": method,
                    "status": str(method_info.get("status", "")).strip(),
                    "note": str(method_info.get("note", "")).strip() or "no_model_csvs",
                    "qrels_lines": 0,
                    "run_lines": 0,
                    "qrels_path": "",
                    "run_path": "",
                    "debug_path": "",
                    "csv_paths": [],
                    "nugget_csv_path": "",
                }
            )
            continue

        method_tag = _safe_name(method)
        nugget_csv_path = out_dir / f"{cluster_name}_{method_tag}_nuggets_dedup.csv"
        nugget_csv_rows = _write_method_dedup_csv(nugget_csv_path, method_csv_paths, method_model_ids)
        query_method_counts[-1]["nugget_csv_path"] = str(nugget_csv_path.resolve())
        qrels_rows, run_rows, debug = qrels_builder(
            query_maps,
            method_csv_paths,
            subtopic=subtopic,
            model=rerank_model,
            emit_match_report=False,
        )

        qrels_path = out_dir / f"{cluster_name}_{method_tag}_real_subtopic.qrels"
        run_path = out_dir / f"{cluster_name}_{method_tag}_real_initial.run"
        debug_path = out_dir / f"{cluster_name}_{method_tag}_query_csv_match_debug.json"
        _write_qrels(qrels_path, qrels_rows)
        _write_run(run_path, run_rows)
        _save_json(
            debug_path,
            {
                "method": method,
                "queries": debug,
                "per_model_csv_paths": [str(p.resolve()) for p in method_csv_paths],
            },
        )
        method_runs.append(
            {
                "method": method,
                "status": str(method_info.get("status", "")).strip(),
                "note": str(method_info.get("note", "")).strip(),
                "qrels_lines": len(qrels_rows),
                "run_lines": len(run_rows),
                "qrels_path": str(qrels_path.resolve()),
                "run_path": str(run_path.resolve()),
                "debug_path": str(debug_path.resolve()),
                "csv_paths": [str(p.resolve()) for p in method_csv_paths],
                "nugget_csv_path": str(nugget_csv_path.resolve()),
                "nugget_csv_rows": nugget_csv_rows,
            }
        )

    return {
        "cluster": cluster_name,
        "job_id": job_id,
        "models": model_ids,
        "method_model_sets": method_model_sets,
        "card_outputs": card_outputs,
        "per_model_csv_paths": [str(p.resolve()) for p in per_model_csv_paths],
        "existing_skipped": len(skipped_ids),
        "newly_created": len(new_outputs),
        "match_build": match_build,
        "query_headers": query_headers,
        "card_rows": card_rows,
        "query_method_counts": query_method_counts,
        "methods": method_runs,
    }


def _evaluate_cluster(run_path: Path, qrels_path: Path, *, cutoff: int, alpha: float, per_query: bool) -> dict[str, Any]:
    run = load_run(run_path)
    qrels = load_subtopic_qrels(qrels_path)
    if not run or not qrels:
        return {
            "skipped": True,
            "reason": "empty_run_or_qrels",
            "run_rows": len(run),
            "qrels_rows": len(qrels),
        }

    try:
        import pyndeval
    except ImportError as e:
        return {
            "skipped": True,
            "reason": "pyndeval_missing",
            "run_rows": len(run),
            "qrels_rows": len(qrels),
            "hint": f"pip install pyndeval ({e})",
        }

    by_query = pyndeval.ndeval(qrels, run, measures=[f"alpha-nDCG@{cutoff}", f"strec@{cutoff}"], alpha=alpha)
    alpha_key = f"alpha-nDCG@{cutoff}"
    strec_key = f"strec@{cutoff}"
    result: dict[str, Any] = {
        "skipped": False,
        "cutoff": cutoff,
        "alpha": alpha,
        "alpha_nDCG": mean(row[alpha_key] for row in by_query.values()),
        "strec": mean(row[strec_key] for row in by_query.values()),
    }
    if per_query:
        result["per_query"] = {
            qid: {"alpha_nDCG": row[alpha_key], "strec": row[strec_key]}
            for qid, row in sorted(by_query.items())
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One job from batch JSON: card2nugget (reuses existing per-model CSVs automatically) -> query2nugget -> per-method qrels/run -> eval.",
    )
    parser.add_argument("--model", default=None, help="OpenAI model override for query2nugget.")
    parser.add_argument(
        "--llm-mode",
        default="batch",
        choices=["batch", "iter"],
        help="OpenAI Batch API vs sync chat per item (card2nugget + query2nugget).",
    )
    parser.add_argument(
        "--match-build",
        default="structured",
        choices=["structured", "llm_rerank"],
        help="qrels/run construction: structured=filters+row match; llm_rerank=second-stage LLM picks rows (needs OPENAI_API_KEY).",
    )
    parser.add_argument("--subtopic", default="1", help="Subtopic id written into qrels.")
    parser.add_argument("--eval-cutoff", type=int, default=20, help="Cutoff for evaluate_pyndeval metrics.")
    parser.add_argument("--eval-alpha", type=float, default=0.5, help="Alpha for alpha-nDCG.")
    parser.add_argument("--eval-per-query", action="store_true", help="Include per-query eval metrics.")
    parser.add_argument(
        "--jobs-json",
        "--jobs-batch-json",
        dest="jobs_batch_json",
        required=True,
        metavar="PATH",
        help="batch_runs JSON (list): entries with job_id, query, and integration_model_search.<method>.model_ids.",
    )
    parser.add_argument("--job-id", required=True, help="Exactly one job_id to run from the JSON (one invocation = one job).")

    parser.add_argument("--output-dir", default=str(PIPELINE_DIR), help=f"Output directory (default: {PIPELINE_DIR})")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    jobs_path = Path(args.jobs_batch_json)
    if not jobs_path.is_file():
        parser.error(f"--jobs-json not found: {jobs_path}")

    job_id_arg = str(args.job_id).strip()
    if not job_id_arg:
        parser.error("--job-id must be non-empty")

    job_sets = _extract_job_sets(jobs_path, {job_id_arg})
    if not job_sets:
        parser.error(f"No valid job entry for job_id={job_id_arg!r} in {jobs_path}")
    if len(job_sets) > 1:
        parser.error(f"Multiple JSON list entries for job_id={job_id_arg!r}; expected exactly one.")

    item = job_sets[0]
    query = item["query"]
    model_ids = item["model_ids"]
    method_model_sets = item.get("method_model_sets", [])
    job_id = str(item["job_id"]).strip() or job_id_arg
    cluster_name = _safe_name(job_id)

    print(f"[jobs] job_id={job_id} | cluster_dir={cluster_name} | models={len(model_ids)}")
    query_maps = map_queries([query], model=args.model, llm_mode=args.llm_mode)
    cluster_out_dir = out_dir / cluster_name
    cluster_out_dir.mkdir(parents=True, exist_ok=True)
    match_log_md = cluster_out_dir / "pipeline_match_log.md"
    summary_path = cluster_out_dir / "pipeline_summary.json"
    summary = _run_cluster(
        cluster_name,
        job_id,
        model_ids,
        method_model_sets,
        query_maps,
        cluster_out_dir,
        str(args.subtopic),
        llm_mode=args.llm_mode,
        match_build=args.match_build,
        rerank_model=args.model,
    )
    summary["query"] = query
    summaries.append(summary)

    # Open-world setting: keep qrels/run artifacts for inspection, but do not compute alpha-nDCG/strec here.
    # The markdown summary below reports matched nugget counts instead of fixed-ground-truth coverage metrics.

    _save_json(summary_path, {"clusters": summaries})
    print(f"[done] summary -> {summary_path.resolve()}")

    summary = summaries[0]
    match_log_md.write_text(
        _format_pipeline_match_markdown(
            jobs_path=jobs_path,
            job_id=job_id,
            query=query,
            card_rows=summary.get("card_rows", []),
            query_headers=summary.get("query_headers", []),
            query_method_counts=summary.get("query_method_counts", []),
        ),
        encoding="utf-8",
    )

    print(f"[done] query2nugget tables -> {match_log_md.resolve()}")


if __name__ == "__main__":
    main()
