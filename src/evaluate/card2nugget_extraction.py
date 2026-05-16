#!/usr/bin/env python3
"""HF model card (full text) → one LLM call → parse markdown table → CSV."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Literal

import duckdb
import yaml
from dotenv import load_dotenv

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
load_dotenv(os.path.join(_repo_root, ".env"), override=False)

from src.config import CARD_CONTENT_RAW, OUTPUT_DIR
from src.evaluate.nugget_schema import NUGGET_SCHEMA_HEADERS
from src.llm.batch import main_batch_query
from src.llm.model import query_openai, setup_openai

SHARD_GLOB = "train-0000*-of-00006.parquet"
EVAL_DIR = Path(OUTPUT_DIR) / "evaluate"
BATCH_DIR = EVAL_DIR / "batch"
CARD2NUGGET_DIR = Path(OUTPUT_DIR) / "card2nugget"
PROMPT_PATH = Path("src/evaluate/card2nugget_prompts.yaml")
PROMPT_KEY = "nugget_schema_mapping"
TEXT_EXTRACTION_MODEL = os.getenv("MODELSEARCHDEMO_TEXT_EXTRACTION_MODEL", "gpt-5.4-mini")
CARD_MAX_CHARS = int(os.getenv("MODELSEARCHDEMO_CARD_MAX_CHARS", "100000"))

OUTPUT_HEADERS = list(NUGGET_SCHEMA_HEADERS)


def _norm(v: Any) -> str:
    if v is None:
        return ""
    text = str(v).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return re.sub(r"\s+", " ", text)


def _read_card_by_model_id(model_id: str) -> str:
    model_id = _norm(model_id)
    if not model_id:
        raise ValueError("model_id is required")
    if not Path(CARD_CONTENT_RAW).exists():
        raise FileNotFoundError(f"CARD_CONTENT_RAW does not exist: {CARD_CONTENT_RAW}")
    parquet_glob = str(Path(CARD_CONTENT_RAW) / SHARD_GLOB)
    query = """
        SELECT card FROM read_parquet(?, union_by_name=true)
        WHERE CAST(modelId AS VARCHAR) = ? LIMIT 1
    """
    with duckdb.connect(":memory:") as con:
        row = con.execute(query, [parquet_glob, model_id]).fetchone()
    if not row or row[0] is None:
        raise FileNotFoundError(f"No card found for modelId={model_id!r}")
    return str(row[0])


def _remove_citation_part(text: str) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = re.sub(r"(?is)```(?:bibtex|latex)?\s*.*?```", "", text)
    lines = cleaned.splitlines()
    kept: list[str] = []
    skipping = False
    for raw_line in lines:
        line = raw_line.strip()
        if re.match(r"(?i)^\s*(#{1,6}\s*)?(citation|cite|references|reference|bibliography)\b", line):
            break
        if re.match(
            r"(?i)^@(article|inproceedings|misc|techreport|phdthesis|mastersthesis|book|incollection|conference|software)\b",
            line,
        ):
            skipping = True
            continue
        if skipping:
            if not line:
                skipping = False
            continue
        kept.append(raw_line)
    return "\n".join(kept)


def _front_matter(text: str) -> str:
    src = text or ""
    if not src.startswith("---"):
        return ""
    end = src.find("\n---", 3)
    return src[3:end] if end != -1 else ""


def _metadata_values(front: str, key: str) -> list[str]:
    lines = front.splitlines()
    values: list[str] = []
    collecting = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^[A-Za-z0-9_-]+:", stripped):
            collecting = False
        if stripped.startswith(f"{key}:"):
            collecting = True
            rest = stripped.split(":", 1)[1].strip()
            if rest:
                values.append(rest.strip("'\""))
            continue
        if collecting and stripped.startswith("- "):
            values.append(stripped[2:].strip().strip("'\""))
    return [v for v in values if v]


def _metadata_lineage(card_text: str) -> tuple[str, str]:
    front = _front_matter(card_text)
    base_values = _metadata_values(front, "base_model")
    base_model = base_values[0] if base_values else ""
    hay = " ".join(
        _metadata_values(front, "tags")
        + _metadata_values(front, "library_name")
        + _metadata_values(front, "base_model_relation")
    ).lower()
    if "quantized" in hay or "gguf" in hay:
        variant = "quantized"
    elif "peft" in hay or "lora" in hay or "adapter" in hay or "adaptor" in hay:
        variant = "adaptor"
    elif "merge" in hay or "merged" in hay:
        variant = "merged"
    elif "finetuned" in hay or "fine-tuned" in hay:
        variant = "finetuned"
    else:
        variant = ""
    return base_model, variant


def _load_prompt_template() -> str:
    if not PROMPT_PATH.exists():
        return ""
    data = yaml.safe_load(PROMPT_PATH.read_text(encoding="utf-8")) or {}
    return str(data.get(PROMPT_KEY, "")).strip()


def _safe_model_id(model_id: str) -> str:
    return _norm(model_id).replace("/", "__")


def _output_paths_for_model(model_id: str, use_batch_dir: bool = False) -> tuple[Path, Path]:
    base = _safe_model_id(model_id)
    _ = use_batch_dir
    out_dir = CARD2NUGGET_DIR
    return out_dir / f"{base}.csv", out_dir / f"{base}_meta.yaml"


def _split_pipe_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [_norm(p) for p in s.split("|")]


def _is_markdown_separator_row(line: str) -> bool:
    cells = _split_pipe_row(line)
    if len(cells) < 2:
        return False
    return all(bool(re.fullmatch(r":?-{2,}:?", c.strip())) for c in cells if c.strip())


def _parse_markdown_table(response_text: str) -> list[dict[str, str]]:
    text = response_text or ""
    bodies: list[str] = []
    for m in re.finditer(r"```(?:markdown|md)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE):
        bodies.append(m.group(1).strip())
    if not bodies and "|" in text:
        bodies.append(text.strip())

    rows_out: list[dict[str, str]] = []
    for body in bodies:
        lines = [ln.strip() for ln in body.splitlines() if "|" in ln and ln.strip()]
        if len(lines) < 3:
            continue
        header_cells = _split_pipe_row(lines[0])
        header_map: dict[str, int] = {}
        for i, raw_h in enumerate(header_cells):
            key = raw_h.lower().replace(" ", "_")
            header_map[key] = i
        start = 1
        if start < len(lines) and _is_markdown_separator_row(lines[start]):
            start += 1
        for ln in lines[start:]:
            if _is_markdown_separator_row(ln):
                continue
            cells = _split_pipe_row(ln)
            if len(header_cells) == len(OUTPUT_HEADERS) and len(cells) == 4:
                cells = [cells[0], "", "", cells[1], cells[2], cells[3]]
            elif len(header_cells) == len(OUTPUT_HEADERS) and len(cells) == 3:
                cells = [cells[0], "", "", "", cells[1], cells[2]]
            row: dict[str, str] = {}
            for h in OUTPUT_HEADERS:
                idx = header_map.get(h.lower(), -1)
                row[h] = cells[idx] if idx >= 0 and idx < len(cells) else ""
            if any(_norm(v) for v in row.values()):
                rows_out.append(row)
        if rows_out:
            break
    return rows_out


def _blank_nugget_row() -> dict[str, str]:
    return {h: "" for h in OUTPUT_HEADERS}


def _is_aggregate_dataset_label(value: str) -> bool:
    s = _norm(value).lower()
    if not s:
        return False
    if re.search(r"\b(average|overall|total|aggregate|summary|recovery)\b", s):
        return True
    return s.startswith("openllm")


def _compact_evidence_text(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").lower())


def _metric_value_has_evidence(value: str, card_text: str) -> bool:
    raw_value = _norm(value)
    if not raw_value:
        return False
    hay = _compact_evidence_text(card_text)
    candidates = _metric_value_candidates(raw_value)
    return any(_compact_evidence_text(c).rstrip("0").rstrip(".") in hay for c in candidates if c)


def _metric_value_candidates(value: str) -> set[str]:
    raw_value = _norm(value)
    out = {raw_value} if raw_value else set()
    try:
        num = float(raw_value)
    except ValueError:
        return out
    if 0 < num <= 1:
        pct = num * 100
        out.update({f"{pct:.1f}", f"{pct:.2f}", f"{pct:.1f}%", f"{pct:.2f}%"})
    return out


def _metric_value_not_from_other_model_link(value: str, card_text: str, model_id: str) -> bool:
    candidates = [_compact_evidence_text(c).rstrip("0").rstrip(".") for c in _metric_value_candidates(value)]
    candidates = [c for c in candidates if c]
    if not candidates:
        return True
    target = _compact_evidence_text(model_id)
    lines = [line for line in (card_text or "").splitlines() if "huggingface.co/" in line.lower()]
    if not lines:
        return True
    for line in lines:
        compact = _compact_evidence_text(line)
        if target in compact and any(c in compact for c in candidates):
            return True
    for line in lines:
        compact = _compact_evidence_text(line)
        if target not in compact and any(c in compact for c in candidates):
            return False
    return True


def _normalize_rows_with_metadata(rows: list[dict[str, str]], model_id: str, card_text: str) -> list[dict[str, str]]:
    base_model, variant = _metadata_lineage(card_text)
    out: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    has_lineage = False
    for row in rows:
        clean = {h: _norm(row.get(h, "")) for h in OUTPUT_HEADERS}
        if not any(clean.values()):
            continue
        clean["Model"] = model_id
        is_lineage = bool(clean["Base_model"] or clean["Model_variant_type"]) and not any(
            clean[h] for h in ("Dataset", "Metric_name", "Metric_value")
        )
        if is_lineage:
            clean["Base_model"] = base_model
            clean["Model_variant_type"] = variant
            if not clean["Base_model"] and not clean["Model_variant_type"]:
                continue
            has_lineage = True
        else:
            clean["Base_model"] = ""
            clean["Model_variant_type"] = ""
        if clean["Metric_name"] and not clean["Metric_value"]:
            clean["Metric_name"] = ""
        if clean["Metric_value"] and not _metric_value_has_evidence(clean["Metric_value"], card_text):
            if clean["Dataset"]:
                clean["Metric_name"] = ""
                clean["Metric_value"] = ""
            else:
                continue
        if clean["Dataset"] and clean["Metric_value"] and not _metric_value_not_from_other_model_link(clean["Metric_value"], card_text, model_id):
            clean["Metric_name"] = ""
            clean["Metric_value"] = ""
        if clean["Dataset"] and _is_aggregate_dataset_label(clean["Dataset"]):
            continue
        if any(clean[h] for h in OUTPUT_HEADERS if h != "Model"):
            sig = tuple(clean[h] for h in OUTPUT_HEADERS)
            if sig in seen:
                continue
            seen.add(sig)
            out.append(clean)
    if (base_model or variant) and not has_lineage:
        lineage = _blank_nugget_row()
        lineage["Model"] = model_id
        lineage["Base_model"] = base_model
        lineage["Model_variant_type"] = variant
        sig = tuple(lineage[h] for h in OUTPUT_HEADERS)
        if sig not in seen:
            out.insert(0, lineage)
    return out


def _write_llm_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(OUTPUT_HEADERS)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def _prepare_model_input(model_id: str) -> dict[str, Any]:
    card_raw = _read_card_by_model_id(model_id)
    card_clean = _remove_citation_part(card_raw)
    card_for_hparam = _norm(card_clean)
    truncated = False
    card_in_prompt = card_for_hparam
    if CARD_MAX_CHARS > 0 and len(card_in_prompt) > CARD_MAX_CHARS:
        card_in_prompt = card_in_prompt[:CARD_MAX_CHARS] + "\n\n[TRUNCATED: card exceeded MODELSEARCHDEMO_CARD_MAX_CHARS]"
        truncated = True

    template = _load_prompt_template()
    if not template:
        raise RuntimeError(f"Missing prompt key {PROMPT_KEY} in {PROMPT_PATH}")
    prompt = template.replace("[[MODEL_CARD]]", card_in_prompt).replace("[[TARGET_MODEL_ID]]", model_id)
    return {
        "model_id": model_id,
        "prompt_key": PROMPT_KEY,
        "prompt": prompt,
        "card_clean": card_clean,
        "card_for_hparam": card_for_hparam,
        "card_truncated": truncated,
        "card_chars_total": len(card_for_hparam),
        "card_chars_in_prompt": len(card_in_prompt),
    }


def _save_model_outputs(prepared: dict[str, Any], response: str, note: str, use_batch_dir: bool = False) -> tuple[Path, Path]:
    csv_path, meta_path = _output_paths_for_model(prepared["model_id"], use_batch_dir=use_batch_dir)
    parsed_rows = _parse_markdown_table(response)
    parsed_rows = _normalize_rows_with_metadata(parsed_rows, prepared["model_id"], prepared["card_clean"])
    _write_llm_csv(csv_path, parsed_rows)
    meta = {
        "modelId": prepared["model_id"],
        "llm_model": TEXT_EXTRACTION_MODEL,
        "prompt_key": prepared["prompt_key"],
        "llm_csv": str(csv_path.resolve()),
        "prompt": prepared["prompt"],
        "raw_response": response,
        "parsed_rows": len(parsed_rows),
        "card_truncated": prepared.get("card_truncated", False),
        "card_chars_total": prepared.get("card_chars_total", 0),
        "card_chars_in_prompt": prepared.get("card_chars_in_prompt", 0),
        "uses_local_table_files": False,
        "llm_turns": [{"turn_index": 1, "note": note}],
    }
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False, allow_unicode=True)
    return csv_path, meta_path


def _extract_text_from_batch_line(item: dict[str, Any]) -> tuple[str, str]:
    resp = item.get("response") or {}
    body = resp.get("body") if isinstance(resp, dict) else {}
    if isinstance(body, dict):
        choices = body.get("choices") or []
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message") or {}
            content = message.get("content", "")
            if isinstance(content, str):
                return content, ""
            if isinstance(content, list):
                parts = []
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        parts.append(str(c.get("text", "")))
                return "\n".join(parts), ""
    err = item.get("error")
    return "", _norm(err) or "batch_output_parse_error"


def _run_batch_query(prepared_items: list[dict[str, Any]]) -> dict[str, tuple[str, str]]:
    ts = int(time.time())
    input_path = BATCH_DIR / f"batch_input_{ts}.jsonl"
    output_path = BATCH_DIR / f"batch_output_{ts}.jsonl"
    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    with open(input_path, "w", encoding="utf-8") as f:
        for item in prepared_items:
            body: dict[str, Any] = {
                "model": TEXT_EXTRACTION_MODEL,
                "messages": [{"role": "user", "content": item["prompt"]}],
                "max_completion_tokens": 4096,
            }
            payload = {
                "custom_id": item["model_id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    main_batch_query(str(input_path), str(output_path))

    out: dict[str, tuple[str, str]] = {}
    if not output_path.exists():
        for item in prepared_items:
            out[item["model_id"]] = ("", "batch_no_output_file")
        return out
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            custom_id = _norm(obj.get("custom_id", ""))
            if not custom_id:
                continue
            text, note = _extract_text_from_batch_line(obj)
            out[custom_id] = (text, note)
    for item in prepared_items:
        if item["model_id"] not in out:
            out[item["model_id"]] = ("", "batch_missing_custom_id")
    return out


def _iter_openai_responses_for_cards(prepared_items: list[dict[str, Any]]) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    if not os.getenv("OPENAI_API_KEY"):
        for item in prepared_items:
            out[item["model_id"]] = ("", "skip_no_api_key")
        return out
    setup_openai("", mode="openai")
    for item in prepared_items:
        mid = item["model_id"]
        try:
            resp = query_openai(item["prompt"], mode="openai", model=TEXT_EXTRACTION_MODEL, max_tokens=4096) or ""
            out[mid] = (resp, "")
        except Exception as e:
            out[mid] = ("", str(e))
    return out


def collect_card2nugget_llm_responses(
    prepared_items: list[dict[str, Any]],
    llm_mode: Literal["batch", "iter"],
) -> dict[str, tuple[str, str]]:
    if llm_mode == "iter":
        print("[card2nugget] llm_mode=iter (sync chat per model)")
        return _iter_openai_responses_for_cards(prepared_items)
    print("[card2nugget] llm_mode=batch (OpenAI Batch API)")
    return _run_batch_query(prepared_items)


def read_llm_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def run_single(model_id: str) -> dict[str, Any]:
    t0 = time.time()
    t_last = t0

    def _tick(stage: str) -> None:
        nonlocal t_last
        now = time.time()
        print(f"[stage] {stage} | +{now - t_last:.3f}s | total={now - t0:.3f}s")
        t_last = now

    print(f"[start] model_id={model_id}")
    prepared = _prepare_model_input(model_id)
    print(f"[prompt] key={prepared['prompt_key']} card_chars={prepared['card_chars_in_prompt']} truncated={prepared['card_truncated']}")
    _tick("prepare_prompt_input")

    response = ""
    note = ""
    if not os.getenv("OPENAI_API_KEY"):
        note = "skip_no_api_key"
        _tick("skip_llm_no_api_key")
    else:
        setup_openai("", mode="openai")
        _tick("setup_openai")
        print("[llm] querying openai...")
        response = query_openai(prepared["prompt"], mode="openai", model=TEXT_EXTRACTION_MODEL, max_tokens=4096) or ""
        _tick(f"query_openai(response_chars={len(response)})")

    csv_path, meta_path = _save_model_outputs(prepared, response, note, use_batch_dir=True)
    _tick("save_csv_and_meta")
    return {"csv_path": str(csv_path.resolve()), "meta_path": str(meta_path.resolve())}


def run_batch(
    model_ids: list[str],
    *,
    llm_mode: Literal["batch", "iter"] = "batch",
) -> list[dict[str, str]]:
    clean_ids = [_norm(m) for m in model_ids if _norm(m)]
    if not clean_ids:
        return []
    print(f"[batch] preparing {len(clean_ids)} model prompts...")
    prepared_items = [_prepare_model_input(m) for m in clean_ids]
    response_map = collect_card2nugget_llm_responses(prepared_items, llm_mode)
    outputs: list[dict[str, str]] = []
    for item in prepared_items:
        model_id = item["model_id"]
        response, note = response_map.get(model_id, ("", "batch_missing_custom_id"))
        csv_path, meta_path = _save_model_outputs(item, response, note, use_batch_dir=True)
        outputs.append({"model_id": model_id, "csv_path": str(csv_path.resolve()), "meta_path": str(meta_path.resolve()), "note": note})
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Model card extraction: full card text → LLM → CSV (single or batch).")
    parser.add_argument("--model-id", default=None, help="Single Hugging Face model id.")
    parser.add_argument("--model-ids", nargs="*", default=None, help="Multiple model ids for batch mode.")
    parser.add_argument("--model-ids-file", default=None, help="Text file with one model id per line.")
    parser.add_argument("--llm-mode", choices=["batch", "iter"], default="batch", help="batch=OpenAI Batch API; iter=sync chat per model (when Batch is slow/stuck).")
    parser.add_argument("--read-csv", nargs="?", const=str(CARD2NUGGET_DIR / "single_modelcard_llm.csv"), metavar="PATH", help="Print stats for parsed tuple CSV.")
    args = parser.parse_args()

    if args.read_csv is not None:
        p = Path(args.read_csv)
        if not p.is_file():
            print(f"missing: {p.resolve()}")
            return
        rows = read_llm_csv(p)
        if not rows:
            print(f"empty: {p.resolve()}")
            return
        headers = list(rows[0].keys())
        print(f"csv = {p.resolve()}")
        print(f"rows = {len(rows)}")
        print(f"headers = {headers}")
        return

    model_ids: list[str] = []
    if args.model_id:
        model_ids.append(args.model_id)
    if args.model_ids:
        model_ids.extend(args.model_ids)
    if args.model_ids_file:
        p = Path(args.model_ids_file)
        if not p.is_file():
            parser.error(f"--model-ids-file not found: {p}")
        with open(p, encoding="utf-8") as f:
            model_ids.extend([line.strip() for line in f if line.strip()])
    model_ids = [_norm(m) for m in model_ids if _norm(m)]
    if not model_ids:
        parser.error("Provide --model-id, or --model-ids, or --model-ids-file")

    if len(model_ids) == 1 and args.llm_mode == "iter":
        result = run_single(model_ids[0])
        print(f"saved_csv = {result['csv_path']}")
        print(f"saved_meta_yaml = {result['meta_path']}")
    else:
        outputs = run_batch(model_ids, llm_mode=args.llm_mode)
        if len(model_ids) == 1:
            out = outputs[0]
            print(f"saved_csv = {out['csv_path']}")
            print(f"saved_meta_yaml = {out['meta_path']}")
            print(f"note = {out.get('note') or 'ok'}")
        else:
            print(f"[batch] saved models = {len(outputs)}")
            for out in outputs:
                print(f"model_id = {out['model_id']}  |  csv = {out['csv_path']}  |  meta = {out['meta_path']}  |  note = {out['note'] or 'ok'}")


if __name__ == "__main__":
    main()
