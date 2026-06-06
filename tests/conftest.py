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
    """呼び出しごとに異なるステムファイルを返す SeparateFn スタブを生成するファクトリ。

    各引数がひとつのステージに対応するファイル名リスト。呼び出し順に対応するステージの
    パスを返す。ステージ数を超えた呼び出しは最後のステージのパスを繰り返す。

    Usage（1段階）:
        fn = make_separate_fn(["lead.wav", "chorus.wav"])

    Usage（2段階）:
        fn = make_separate_fn(
            ["song_Vocals.wav", "song_Instrumental.wav"],  # Stage1
            ["song_lead.wav",   "song_chorus.wav"],        # Stage2
        )
    """

    def factory(*stages: list[str]):
        all_stages: list[tuple[Path, ...]] = []
        for stage_names in stages:
            paths: list[Path] = []
            for name in stage_names:
                p = tmp_path / name
                p.write_bytes(b"FAKE_AUDIO")
                paths.append(p)
            all_stages.append(tuple(paths))

        call_index = 0

        def stub(
            separator: object,
            model_filename: str,
            input_path: Path,
            output_names: dict[str, str] | None = None,
        ) -> SeparationResult:
            nonlocal call_index
            idx = min(call_index, len(all_stages) - 1)
            call_index += 1
            return SeparationResult(
                input_path=input_path,
                output_paths=all_stages[idx],
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
