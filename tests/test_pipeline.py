"""pipeline.py のユニットテスト。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from chorus_extraction.config import build_run_config
from chorus_extraction.errors import SeparationError
from chorus_extraction.pipeline import extract, separate_vocal
from chorus_extraction.separator_runner import SeparationResult


def _make_cfg(tmp_path: Path, mode: str = "vocal"):
    with patch("chorus_extraction.config._check_cuda", return_value=False):
        return build_run_config(
            inputs=[tmp_path / "vocals.wav"],
            mode=mode,  # type: ignore[arg-type]
            output_dir=tmp_path / "out",
            output_format="wav",
            stage1_model_name=None,
            stage2_model_name=None,
            model_dir=tmp_path / "models",
            keep_intermediate=False,
            device="cpu",
            lead_name_template="{stem}_lead",
            chorus_name_template="{stem}_chorus",
            verbosity=0,
        )


class TestSeparateVocal:
    def test_returns_extraction_result(self, tmp_path: Path, mock_separator, make_separate_fn, sample_wav: Path) -> None:
        lead_file = tmp_path / "out" / "vocals_lead.wav"
        chorus_file = tmp_path / "out" / "vocals_chorus.wav"
        lead_file.parent.mkdir(parents=True, exist_ok=True)
        lead_file.write_bytes(b"LEAD")
        chorus_file.write_bytes(b"CHORUS")

        separate_fn = make_separate_fn(["vocals_lead.wav", "vocals_chorus.wav"])
        cfg = _make_cfg(tmp_path, "vocal")

        result = separate_vocal(
            sample_wav,
            cfg,
            separator=mock_separator,
            separate_fn=separate_fn,
        )

        assert result.input_path == sample_wav
        assert len(result.all_output_paths) == 2


class TestExtract:
    def test_vocal_mode(self, tmp_path: Path, mock_separator, make_separate_fn, sample_wav: Path) -> None:
        separate_fn = make_separate_fn(["sample_lead.wav", "sample_chorus.wav"])
        with patch("chorus_extraction.config._check_cuda", return_value=False):
            cfg = build_run_config(
                inputs=[sample_wav],
                mode="vocal",
                output_dir=tmp_path / "out",
                output_format="wav",
                stage1_model_name=None,
                stage2_model_name=None,
                model_dir=tmp_path / "models",
                keep_intermediate=False,
                device="cpu",
                lead_name_template="{stem}_lead",
                chorus_name_template="{stem}_chorus",
                verbosity=0,
            )

        results = extract(cfg, separator=mock_separator, separate_fn=separate_fn)
        assert len(results) == 1
        assert results[0].input_path == sample_wav

    def test_invalid_input_raises(self, tmp_path: Path, mock_separator, make_separate_fn) -> None:
        from chorus_extraction.errors import InvalidInputError

        separate_fn = make_separate_fn([])
        with patch("chorus_extraction.config._check_cuda", return_value=False):
            cfg = build_run_config(
                inputs=[tmp_path / "nonexistent.wav"],
                mode="vocal",
                output_dir=tmp_path / "out",
                output_format="wav",
                stage1_model_name=None,
                stage2_model_name=None,
                model_dir=tmp_path / "models",
                keep_intermediate=False,
                device="cpu",
                lead_name_template="{stem}_lead",
                chorus_name_template="{stem}_chorus",
                verbosity=0,
            )

        with pytest.raises(InvalidInputError):
            extract(cfg, separator=mock_separator, separate_fn=separate_fn)
