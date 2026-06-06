"""入力検証・出力命名・中間ファイル管理。"""

from __future__ import annotations

import contextlib
import tempfile
from collections.abc import Iterator
from pathlib import Path

from chorus_extraction.config import SUPPORTED_INPUT_EXTENSIONS, RunConfig
from chorus_extraction.errors import InvalidInputError


def validate_input(path: Path) -> None:
    """音声入力ファイルの事前検証。パスを正規化してシンボリックリンク・トラバーサルを解決する。

    Raises:
        InvalidInputError: ファイルが存在しない、空、または非対応フォーマットの場合。
    """
    path = path.resolve()
    if not path.exists():
        raise InvalidInputError(f"入力ファイルが見つかりません: {path}")
    if not path.is_file():
        raise InvalidInputError(f"入力パスはファイルではありません: {path}")
    if path.stat().st_size == 0:
        raise InvalidInputError(f"入力ファイルが空です: {path}")
    if path.suffix.lower() not in SUPPORTED_INPUT_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_INPUT_EXTENSIONS))
        raise InvalidInputError(
            f"非対応のファイル形式です: {path.suffix!r}"
            f"\n対応フォーマット: {supported}"
        )


def build_output_names(stem: str, cfg: RunConfig) -> dict[str, str]:
    """カラオケモデルの出力ステム名をユーザー指定テンプレートでリネームする dict を返す。

    カラオケ系モデルは 'Vocals'(リード) / 'Instrumental'(ハモリ) の2ステムを出力する。
    テンプレート展開後のファイル名にパストラバーサル（`..` 等）が含まれないことを検証する。
    """
    lead_name = cfg.lead_name_template.replace("{stem}", stem)
    chorus_name = cfg.chorus_name_template.replace("{stem}", stem)
    _validate_output_filename(lead_name, "lead-name")
    _validate_output_filename(chorus_name, "chorus-name")
    return {
        "Vocals": lead_name,
        "Instrumental": chorus_name,
    }


def _validate_output_filename(name: str, option: str) -> None:
    """出力ファイル名にディレクトリ区切り文字や `..` が含まれないことを検証する。"""
    import os

    seps = filter(None, (os.sep, os.altsep, "/", "\\"))
    if any(sep in name for sep in seps):
        raise InvalidInputError(
            f"--{option} にパス区切り文字を含めることはできません: {name!r}"
        )
    if ".." in name.split("/") or ".." in name.split("\\") or name == "..":
        raise InvalidInputError(
            f"--{option} に '..' を含めることはできません: {name!r}"
        )


@contextlib.contextmanager
def intermediate_dir(cfg: RunConfig) -> Iterator[Path]:
    """Stage1 中間ステムの保存先を提供するコンテキストマネージャ。

    keep_intermediate=True の場合は output_dir/intermediate/ を作成して返す（削除しない）。
    False の場合は TemporaryDirectory を使い終了後に自動削除する。
    """
    if cfg.keep_intermediate:
        dest = cfg.output_dir / "intermediate"
        dest.mkdir(parents=True, exist_ok=True)
        yield dest
    else:
        with tempfile.TemporaryDirectory(prefix="chorus_extraction_") as tmp:
            yield Path(tmp)
