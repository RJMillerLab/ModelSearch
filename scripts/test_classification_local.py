#!/usr/bin/env python3
"""
Run classification module tests without loading the full src.search package
(avoids baseline1/card2card deps). Execute from repo root:
  python3 scripts/test_classification_local.py
  or with conda: python scripts/test_classification_local.py
"""
import sys
import os

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(repo_root)
sys.path.insert(0, repo_root)

# Run classification with argv = ['classification', 'test'] so the test branch runs
sys.argv = ["classification", "test"]

# Load and execute classification module as __main__
import runpy
runpy.run_path(
    os.path.join(repo_root, "src", "search", "classification.py"),
    run_name="__main__",
)
