from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

from search_engine import LocalSearchEngine


BUNDLE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BUNDLE_DIR / "data"
DEFAULT_DATASET = DEFAULT_DATA_DIR / "training" / "wikipedia-20231101-en-00000.parquet"


def read_wikipedia_articles(
    dataset_path: Path, limit: int | None = None, batch_size: int = 512
) -> Iterator[tuple[str, str, str]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "Importing Wikipedia requires pyarrow. Install it with "
            "`py -m pip install -r requirements-data.txt`."
        ) from error

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Wikipedia shard not found: {dataset_path}")

    dataset = parquet.ParquetFile(dataset_path)
    imported = 0
    scanned = 0
    for batch in dataset.iter_batches(
        batch_size=batch_size, columns=["id", "title", "text"]
    ):
        page_ids = batch.column("id").to_pylist()
        titles = batch.column("title").to_pylist()
        bodies = batch.column("text").to_pylist()
        for page_id, title, body in zip(page_ids, titles, bodies):
            if not isinstance(title, str) or not isinstance(body, str):
                continue
            clean_title = " ".join(title.split())
            clean_body = " ".join(body.split())
            if not clean_title or len(clean_body) < 80:
                continue
            yield str(page_id), clean_title, clean_body
            imported += 1
            scanned += 1
            if scanned % 10_000 == 0:
                print(f"Read {scanned:,} Wikipedia articles…")
            if limit is not None and imported >= limit:
                return


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import the downloaded English Wikipedia shard into the local search index."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, help="Optional maximum articles (default: import the full shard).")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")

    engine = LocalSearchEngine(args.data_dir)
    try:
        count = engine.import_wikipedia_articles(
            read_wikipedia_articles(args.dataset, args.limit, args.batch_size),
            batch_size=args.batch_size,
        )
        print(
            f"Imported or updated {count:,} Wikipedia articles; "
            f"the search index now contains {engine.document_count():,} documents."
        )
    finally:
        engine.close()


if __name__ == "__main__":
    main()
