"""audio_io.py のユニットテスト。"""

from __future__ import annotations

from pathlib import Path

import pytest

from chorus_extraction.audio_io import build_output_names, validate_input
from chorus_extraction.errors import InvalidInputError


class TestValidateInput:
    def test_valid_wav(self, sample_wav: Path) -> None:
        validate_input(sample_wav)  # 例外が出なければ OK

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(InvalidInputError, match="見つかりません"):
            validate_input(tmp_path / "nonexistent.wav")

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.wav"
        empty.write_bytes(b"")
        with pytest.raises(InvalidInputError, match="空"):
            validate_input(empty)

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        txt = tmp_path / "audio.txt"
        txt.write_bytes(b"hello")
        with pytest.raises(InvalidInputError, match="非対応"):
            validate_input(txt)

    def test_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(InvalidInputError, match="ファイルではありません"):
            validate_input(tmp_path)


class TestBuildOutputNames:
    def test_template_substitution(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        from chorus_extraction.config import build_run_config

        with patch("chorus_extraction.config._check_cuda", return_value=False):
            cfg = build_run_config(
                inputs=[tmp_path / "a.wav"],
                mode="vocal",
                output_dir=tmp_path,
                output_format="wav",
                stage1_model_name=None,
                stage2_model_name=None,
                model_dir=tmp_path,
                keep_intermediate=False,
                device="cpu",
                lead_name_template="{stem}_lead",
                chorus_name_template="{stem}_chorus",
                verbosity=0,
            )

        names = build_output_names("my_song", cfg)
        assert names["Vocals"] == "my_song_lead"
        assert names["Instrumental"] == "my_song_chorus"
