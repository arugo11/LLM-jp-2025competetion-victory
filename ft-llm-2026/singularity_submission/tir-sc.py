# TIR + Self-Consistency の推論コード
from __future__ import annotations

import asyncio

from config import config_from_args, parse_args
from pipeline import run_pipeline


def main() -> None:
    """エントリーポイント."""
    args = parse_args()
    config = config_from_args(args)
    asyncio.run(run_pipeline(config))


if __name__ == "__main__":
    main()