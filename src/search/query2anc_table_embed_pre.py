"""Precompute table dense embeddings for query-to-anchor-table selection."""

from __future__ import annotations

import argparse

from src.search.query2anc_table_embed_builder import Query2AnchorTableEmbeddingBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="Build table dense index (.npz) for embedding anchor selection.")
    parser.add_argument("--resources", nargs="+", default=["hugging"], choices=["hugging", "github", "arxiv"])
    parser.add_argument(
        "--output_npz",
        default="",
        help="Defaults to data_251117/query2anc_table_embeddings[_hugging].npz.",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    resources = [str(r).strip().lower() for r in (args.resources or ["hugging"]) if str(r).strip()]
    Query2AnchorTableEmbeddingBuilder(resources=resources, output_npz=(args.output_npz or None)).build(
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
