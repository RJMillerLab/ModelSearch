#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, List

from src.config import JOBS_DIR


def _load_batch_jobs(jobs_json: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(jobs_json):
        raise FileNotFoundError(f"jobs json not found: {jobs_json}")
    with open(jobs_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("jobs json must be a list")
    jobs: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        job_id = str(item.get("job_id", "")).strip()
        if not job_id:
            continue
        jobs.append(item)
    return jobs


def _has_pipeline_summary(job_id: str) -> bool:
    summary_path = os.path.join(JOBS_DIR, job_id, "evaluate", "pipeline_summary.json")
    return os.path.isfile(summary_path)


def _run_one_job(jobs_json: str, job_id: str, llm_mode: str) -> int:
    cmd = [
        sys.executable,
        "-m",
        "src.evaluate.wrap_card_query_eval",
        "--jobs-json",
        jobs_json,
        "--job-id",
        job_id,
        "--llm-mode",
        llm_mode,
    ]
    print(f"[run] {job_id}")
    print("      " + " ".join(cmd))
    proc = subprocess.run(cmd)
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run wrap_card_query_eval per job from a batch summary JSON. "
            "Default behavior: run only jobs missing evaluate/pipeline_summary.json."
        )
    )
    parser.add_argument(
        "--jobs_json",
        required=True,
        help="Path to batch summary json (for example: jobs_251117/batch_runs/batch_preset_queries_*.json).",
    )
    parser.add_argument(
        "--llm_mode",
        choices=["iter", "batch"],
        default="iter",
        help="llm mode passed to wrap_card_query_eval (default: iter).",
    )
    parser.add_argument(
        "--all_jobs",
        action="store_true",
        help="Run all jobs in the batch summary (default: only missing pipeline_summary).",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop immediately if any job fails (default: continue).",
    )
    args = parser.parse_args()

    jobs_json = os.path.abspath(args.jobs_json)
    jobs = _load_batch_jobs(jobs_json)
    if not jobs:
        print("No valid job_id entries found in jobs_json.")
        return 1

    all_job_ids = [str(item["job_id"]).strip() for item in jobs]
    if args.all_jobs:
        target_job_ids = all_job_ids
    else:
        target_job_ids = [job_id for job_id in all_job_ids if not _has_pipeline_summary(job_id)]

    print(f"Jobs in batch: {len(all_job_ids)}")
    print(f"Target jobs to run: {len(target_job_ids)}")
    if not target_job_ids:
        print("Nothing to run. All jobs already have evaluate/pipeline_summary.json.")
        return 0

    ok = 0
    failed: List[str] = []
    for idx, job_id in enumerate(target_job_ids, start=1):
        print(f"\n[{idx}/{len(target_job_ids)}]")
        code = _run_one_job(jobs_json, job_id, args.llm_mode)
        if code == 0:
            ok += 1
            print(f"[ok] {job_id}")
            continue
        failed.append(job_id)
        print(f"[fail] {job_id} (exit={code})")
        if args.stop_on_error:
            break

    print("\nRerun summary:")
    print(f"  success: {ok}")
    print(f"  failed: {len(failed)}")
    if failed:
        print("  failed_job_ids:")
        for job_id in failed:
            print(f"    - {job_id}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
