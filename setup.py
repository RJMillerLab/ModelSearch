"""
Setup script for ModelSearch package
"""
from setuptools import setup, find_packages

setup(
    name="modelsearch",
    version="0.1.0",
    description="Dense semantic search over Hugging Face model cards",
    author="Zhengyuan Dong",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "duckdb>=0.8.0",
        "pyarrow>=10.0.0",
        "sentence-transformers>=2.2.0",
        "torch>=1.12.0",
        "faiss-cpu>=1.7.4",
        "tqdm>=4.64.0",
    ],
    python_requires=">=3.8",
)

