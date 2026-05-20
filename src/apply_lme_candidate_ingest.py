"""Compatibility entrypoint; see ``pipeline.ingest_candidates``."""

from pipeline.ingest_candidates import main

if __name__ == "__main__":
    raise SystemExit(main())
