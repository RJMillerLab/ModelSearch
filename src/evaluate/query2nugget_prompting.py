"""Prompt construction and response normalization for query2nugget."""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
load_dotenv(os.path.join(_repo_root, ".env"), override=False)

from src.evaluate.nugget_schema import NUGGET_SCHEMA_HEADERS

QUERY_MAP_HEADERS = list(NUGGET_SCHEMA_HEADERS)

PROMPT_PATH = Path("src/evaluate/query2nugget_prompts.yaml")
PROMPT_KEY = "query_to_nugget_headers"
FILTER_PROMPT_KEY = "query_to_nugget_filter"
TEXT_MODEL = os.getenv("MODELSEARCHDEMO_TEXT_EXTRACTION_MODEL", "gpt-5.4-mini")


def _load_prompt_template(prompt_key: str = PROMPT_KEY) -> str:
    if not PROMPT_PATH.exists():
        return ""
    data = yaml.safe_load(PROMPT_PATH.read_text(encoding="utf-8")) or {}
    return str(data.get(prompt_key, "")).strip()


def build_filter_prompt(query: str, header_values: list[dict[str, str]], candidate_lines: list[str]) -> str:
    template = _load_prompt_template(FILTER_PROMPT_KEY)
    if not template:
        raise RuntimeError(f"Missing prompt key {FILTER_PROMPT_KEY} in {PROMPT_PATH}")
    return (
        template.replace("[[USER_QUERY]]", query.strip())
        .replace("[[HEADER_VALUES_JSON]]", json.dumps(header_values, ensure_ascii=False))
        .replace("[[CANDIDATE_ROWS]]", "\n".join(candidate_lines))
    )


def extract_json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```\s*$", "", raw)
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            try:
                obj = json.loads(match.group(0))
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                pass
    return None


def build_prompt_for_query(query: str) -> str:
    template = _load_prompt_template()
    if not template:
        raise RuntimeError(f"Missing prompt key {PROMPT_KEY} in {PROMPT_PATH}")
    headers_yaml = yaml.safe_dump(QUERY_MAP_HEADERS, allow_unicode=True).strip()
    return template.replace("[[HEADERS_YAML]]", headers_yaml).replace("[[USER_QUERY]]", (query or "").strip())


def finalize_map_response(query: str, prompt: str, raw: str) -> dict[str, Any]:
    parsed = extract_json_object(raw)
    if not parsed:
        header_list = _extract_header_list_text(raw)
        if header_list:
            related = [{"header": h, "keywords": []} for h in header_list]
            out = {"query": query, "related": related, "clusters": [{"related": related, "filters": []}], "header_list": header_list}
            out["prompt"] = prompt
            out["raw_response"] = raw
            return out
        return {
            "query": query,
            "related": [],
            "clusters": [],
            "header_list": [],
            "error": "parse_failed",
            "prompt": prompt,
            "raw_response": raw,
        }
    out = _validate_and_normalize(parsed, query)
    out["prompt"] = prompt
    out["raw_response"] = raw
    return out


def normalize_filter_dict(f: dict[str, Any]) -> dict[str, str] | None:
    col = str(f.get("column", "")).strip()
    if col not in NUGGET_SCHEMA_HEADERS:
        return None
    needle = str(f.get("contains", "")).strip() or str(f.get("equals", "")).strip()
    if not needle:
        return None
    return {"column": col, "contains": needle}


def normalize_related_entries(related_raw: list[Any]) -> list[dict[str, Any]]:
    allowed = set(QUERY_MAP_HEADERS)
    out: list[dict[str, Any]] = []
    for item in related_raw:
        if not isinstance(item, dict):
            continue
        header = str(item.get("header", "")).strip()
        if header not in allowed:
            continue
        keywords = item.get("keywords")
        if isinstance(keywords, str):
            keywords = [keywords]
        elif not isinstance(keywords, list):
            keywords = []
        entry: dict[str, Any] = {"header": header, "keywords": [str(x).strip() for x in keywords if str(x).strip()]}
        value_contains = str(item.get("value_contains", "")).strip()
        if value_contains:
            entry["value_contains"] = value_contains
        out.append(entry)
    return out


def cluster_from_related_and_filters(related: list[dict[str, Any]], top_filters: list[dict[str, str]]) -> dict[str, Any]:
    filters = list(top_filters)
    related_clean: list[dict[str, Any]] = []
    for item in related:
        value_contains = str(item.get("value_contains", "")).strip()
        header = str(item.get("header", "")).strip()
        if value_contains and header in NUGGET_SCHEMA_HEADERS:
            filters.append({"column": header, "contains": value_contains})
        if header:
            related_clean.append({"header": header, "keywords": item.get("keywords", [])})
    return {"related": related_clean, "filters": filters}


def _extract_header_list_text(text: str) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    allowed = set(QUERY_MAP_HEADERS)
    out: list[str] = []
    seen: set[str] = set()
    for line in (ln.strip() for ln in raw.splitlines() if ln.strip()):
        s = re.sub(r"^\s*[-*•\d\.)\s]+", "", line).strip()
        if ":" in s:
            s = s.split(":", 1)[0].strip()
        if s in allowed and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _clusters_from_parsed(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    top_filters: list[dict[str, str]] = []
    if isinstance(parsed.get("filters"), list):
        for f in parsed["filters"]:
            if isinstance(f, dict):
                normalized = normalize_filter_dict(f)
                if normalized:
                    top_filters.append(normalized)

    clusters_raw = parsed.get("clusters")
    if isinstance(clusters_raw, list) and clusters_raw:
        clusters_out: list[dict[str, Any]] = []
        for cluster in clusters_raw:
            if not isinstance(cluster, dict):
                continue
            related = normalize_related_entries(cluster.get("related", []) if isinstance(cluster.get("related"), list) else [])
            filters: list[dict[str, str]] = list(top_filters)
            if isinstance(cluster.get("filters"), list):
                for f in cluster["filters"]:
                    if isinstance(f, dict):
                        normalized = normalize_filter_dict(f)
                        if normalized:
                            filters.append(normalized)
            one = cluster_from_related_and_filters(related, filters)
            if one["related"] or one["filters"]:
                clusters_out.append(one)
        if clusters_out:
            return clusters_out

    related = normalize_related_entries(parsed.get("related", []) if isinstance(parsed.get("related"), list) else [])
    one = cluster_from_related_and_filters(related, top_filters)
    if one["related"] or one["filters"]:
        return [one]
    return []


def _validate_and_normalize(parsed: dict[str, Any], user_query: str) -> dict[str, Any]:
    q = str(parsed.get("query", "")).strip() or user_query
    clusters = _clusters_from_parsed(parsed)
    if not clusters:
        selected = parsed.get("selected_headers", [])
        if isinstance(selected, list):
            related = [
                {"header": str(h).strip(), "keywords": []}
                for h in selected
                if str(h).strip() in QUERY_MAP_HEADERS
            ]
            if related:
                clusters = [{"related": related, "filters": []}]

    flat_related: list[dict[str, Any]] = []
    seen_h: set[str] = set()
    for cluster in clusters:
        for item in cluster.get("related", []):
            header = str(item.get("header", "")).strip()
            if header and header not in seen_h:
                seen_h.add(header)
                flat_related.append({"header": header, "keywords": list(item.get("keywords", []))})

    out: dict[str, Any] = {
        "query": q,
        "related": flat_related,
        "clusters": clusters,
        "header_list": [x["header"] for x in flat_related],
    }
    header_values = parsed.get("header_values", [])
    if isinstance(header_values, list):
        out["header_values"] = [
            {"header": str(x.get("header", "")).strip(), "contains": str(x.get("contains", "")).strip()}
            for x in header_values
            if isinstance(x, dict)
            and str(x.get("header", "")).strip() in QUERY_MAP_HEADERS
            and str(x.get("contains", "")).strip()
        ]
        for hv in out["header_values"]:
            header = hv["header"]
            if header not in seen_h:
                seen_h.add(header)
                out["related"].append({"header": header, "keywords": []})
                out["header_list"].append(header)
    return out
