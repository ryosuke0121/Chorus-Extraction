"""共有 fixture。Separator をモックに差し替えてテストを高速化する。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from chorus_extraction.separator_runner import SeparationResult


@pytest.fixture()
def mock_separator() -> MagicMock:
    """audio_separator.separator.Separator のモック。load_model / separate をスタブ化する。"""
    sep = MagicMock()
    sep.load_model.return_value = None
    return sep


@pytest.fixture()
def make_separate_fn(tmp_path: Path):
    """指定したステムファイルを tmp_path に作成して返す SeparateFn スタブを生成するファクトリ。

    Usage:
        separate_fn = make_separate_fn(["song_Vocals.wav", "song_Instrumental.wav"])
    """

    def factory(stem_names: list[str]):
        created: list[Path] = []
        for name in stem_names:
            p = tmp_path / name
            p.write_bytes(b"FAKE_AUDIO")
            created.append(p)

        def stub(
            separator: object,
            model_filename: str,
            input_path: Path,
            output_names: dict[str, str] | None = None,
        ) -> SeparationResult:
            return SeparationResult(
                input_path=input_path,
                output_paths=tuple(created),
            )

        return stub

    return factory


@pytest.fixture()
def sample_wav(tmp_path: Path) -> Path:
    """ダミー音声ファイル（RIFF ヘッダのみ）を返す。"""
    path = tmp_path / "sample.wav"
    # 最小限の RIFF/WAV ヘッダ（44 bytes）
    path.write_bytes(
        b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00"
        b"\x01\x00\x01\x00\x44\xac\x00\x00\x88X\x01\x00"
        b"\x02\x00\x10\x00data\x00\x00\x00\x00"
    )
    return path
