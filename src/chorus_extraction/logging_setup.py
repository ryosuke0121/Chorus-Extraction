"""ロギング設定。verbosity レベルに応じて標準エラーへ出力する。"""

from __future__ import annotations

import logging
import sys


def configure_logging(verbosity: int) -> None:
    """verbosity カウント（-v の回数）に応じてログレベルを設定する。

    0 -> WARNING, 1 -> INFO, 2+ -> DEBUG
    """
    # Windows でシステムロケール非対応文字（例: ⧸）を含むパスを表示できるよう UTF-8 に設定
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

    level_map = {0: logging.WARNING, 1: logging.INFO}
    level = level_map.get(verbosity, logging.DEBUG)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    formatter = logging.Formatter("%(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    if not root_logger.handlers:
        root_logger.addHandler(handler)
