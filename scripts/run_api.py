#!/usr/bin/env python3
"""Run the ICMI API server (repo-local data layer).

Example:
    python scripts/run_api.py            # http://127.0.0.1:8000 (docs at /docs)
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="auto-reload (dev)")
    args = parser.parse_args(argv)

    import uvicorn

    uvicorn.run(
        "icmi.api.app:create_app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        factory=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
