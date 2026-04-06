# ModelSearch Demo

ModelSearch is a demo for model-card retrieval, query-to-table-to-model retrieval, and table integration over retrieved CSV tables. The current codebase is centered on the Flask demo in `src/demo/`, retrieval in `src/search/`, and integration in `src/integration/`.

## Quick Start

1. Download the necessary data and repositories.

- Install Python dependencies from `requirements.txt`.
- Prepare the local data used by [src/config.py](src/config.py), especially ModelTables data.
- Download or link the external repositories used by this project, especially `Blend_internal` and `alite_internal`.
- Install extra retrieval dependencies when needed, especially `faiss` and `pyserini` with Java.

Detailed build and data-prep notes are in [docs/build_index.md](docs/build_index.md).

2. Start the demo.

```bash
python -m src.demo.backend
python -m src.demo.frontend
```

Then open `http://localhost:5001`.

## Acknowledgments

This demo builds on several external tools and codebases:

- `Pyserini` for sparse / Lucene-based retrieval
- `FAISS` for dense vector search
- `Blend` and `Blend_internal` for table search
- `ALITE` and `alite_internal` for table integration
- `ModelTables` for the underlying data pipeline and datasets
