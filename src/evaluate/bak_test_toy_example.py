#!/usr/bin/env python3
from __future__ import annotations

import math
import itertools
import subprocess
import sys
from pathlib import Path


ROOT = Path("/Users/doradong/Repo/CoverageBench")
IR_EVAL = ROOT / "scripts" / "evaluate_ir_measures.py"
PYN_EVAL = ROOT / "scripts" / "evaluate_pyndeval.py"
RUN = ROOT / "scripts" / "data" / "toy_initial.run"
QRELS = ROOT / "scripts" / "data" / "toy_subtopic.qrels"

TOY_RANKING = [
    set(),
    {1, 2},
    set(),
    {2},
    {3, 4},
    {5},
]
TOTAL_NUGGETS = 5
ALPHA = 0.5


def run_eval(cutoff: int) -> None:
    ir_cmd = [
        sys.executable,
        str(IR_EVAL),
        "--run",
        str(RUN),
        "--qrels",
        str(QRELS),
        "--query",
        "toy-q1",
        "--cutoff",
        str(cutoff),
        "--quiet-header",
    ]
    pyn_cmd = [
        sys.executable,
        str(PYN_EVAL),
        "--run",
        str(RUN),
        "--qrels",
        str(QRELS),
        "--query",
        "toy-q1",
        "--cutoff",
        str(cutoff),
        "--quiet-header",
    ]
    ir_result = subprocess.run(ir_cmd, check=True, capture_output=True, text=True)
    pyn_result = subprocess.run(pyn_cmd, check=True, capture_output=True, text=True)
    metrics = {}
    for output in (ir_result.stdout, pyn_result.stdout):
        current = None
        for raw_line in output.splitlines():
            line = raw_line.strip()
            if line == "aggregate":
                current = "aggregate"
                continue
            if current == "aggregate" and ":" in line:
                key, value = line.split(":", 1)
                metrics[key.strip()] = value.strip()
    return metrics


def run_alpha_zero(cutoff: int):
    cmd = [
        sys.executable,
        str(PYN_EVAL),
        "--run",
        str(RUN),
        "--qrels",
        str(QRELS),
        "--query",
        "toy-q1",
        "--cutoff",
        str(cutoff),
        "--alpha",
        "0",
        "--quiet-header",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    metrics = {}
    current = None
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if line == "aggregate":
            current = "aggregate"
            continue
        if current == "aggregate" and ":" in line:
            key, value = line.split(":", 1)
            metrics[key.strip()] = value.strip()
    return metrics


def dcg_exp(gains):
    total = 0.0
    for idx, rel in enumerate(gains, start=1):
        total += (2**rel - 1) / math.log2(idx + 1)
    return total


def ndcg_expected(cutoff: int) -> float:
    rels = [len(nugs) for nugs in TOY_RANKING[:cutoff]]
    dcg = dcg_exp(rels)
    ideal = sorted((len(nugs) for nugs in TOY_RANKING), reverse=True)[:cutoff]
    idcg = dcg_exp(ideal)
    return 0.0 if idcg == 0 else dcg / idcg


def alpha_gain_for_order(order):
    seen = {}
    gains = []
    for nuggets in order:
        gain = 0.0
        for nugget in nuggets:
            prev = seen.get(nugget, 0)
            gain += (1 - ALPHA) ** prev
            seen[nugget] = prev + 1
        gains.append(gain)
    return gains


def alpha_dcg_from_gains(gains):
    total = 0.0
    for idx, gain in enumerate(gains, start=1):
        total += gain / math.log2(idx + 1)
    return total


def alpha_ndcg_expected(cutoff: int) -> float:
    actual_order = TOY_RANKING[:cutoff]
    actual_gains = alpha_gain_for_order(actual_order)
    actual_dcg = alpha_dcg_from_gains(actual_gains)

    best = 0.0
    for perm in itertools.permutations(TOY_RANKING, cutoff):
        gains = alpha_gain_for_order(perm)
        best = max(best, alpha_dcg_from_gains(gains))
    return 0.0 if best == 0 else actual_dcg / best


def strec_expected(cutoff: int) -> float:
    covered = set()
    for nuggets in TOY_RANKING[:cutoff]:
        covered.update(nuggets)
    return len(covered) / TOTAL_NUGGETS


def main() -> None:
    print("Toy setup")
    print("  query: toy-q1")
    print("  rank 1 -> {}")
    print("  rank 2 -> {1,2}")
    print("  rank 3 -> {}")
    print("  rank 4 -> {2}")
    print("  rank 5 -> {3,4}")
    print("  rank 6 -> {5}")
    print("  query: toy-q2")
    print("  rank 1 -> {1,2}")
    print("  rank 2 -> {3}")
    print("  rank 3 -> {}")
    print("  rank 4 -> {4}")
    print()
    demo_cmd = [
        sys.executable,
        str(PYN_EVAL),
        "--run",
        str(RUN),
        "--qrels",
        str(QRELS),
        "--cutoff",
        "4",
        "--per-query",
    ]
    demo_output = subprocess.run(demo_cmd, check=True, capture_output=True, text=True).stdout.strip()
    print("| cutoff | expected nDCG | actual nDCG | alpha-nDCG (alpha=0) | expected alpha-nDCG | actual alpha-nDCG | expected strec | actual strec |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for cutoff in range(1, 7):
        metrics = run_eval(cutoff)
        alpha_zero_metrics = run_alpha_zero(cutoff)
        exp_ndcg = ndcg_expected(cutoff)
        exp_alpha = alpha_ndcg_expected(cutoff)
        exp_strec = strec_expected(cutoff)
        print(
            f"| @{cutoff} | {exp_ndcg:.6f} | {metrics.get(f'nDCG@{cutoff}', 'NA')} | "
            f"{alpha_zero_metrics.get(f'alpha-nDCG@{cutoff}', 'NA')} | "
            f"{exp_alpha:.6f} | {metrics.get(f'alpha-nDCG@{cutoff}', 'NA')} | "
            f"{exp_strec:.6f} | {metrics.get(f'strec@{cutoff}', 'NA')} |"
        )
    print()
    print("Aggregate vs per-query demo")
    print(f"$ {' '.join(demo_cmd)}")
    print(demo_output)


if __name__ == "__main__":
    main()
