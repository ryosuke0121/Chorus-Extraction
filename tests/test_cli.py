"""cli.py の CLI テスト。Typer CliRunner を使用。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from chorus_extraction.cli import app

runner = CliRunner()


class TestListModels:
    def test_list_models_exits_zero(self) -> None:
        result = runner.invoke(app, ["--list-models"])
        assert result.exit_code == 0
        assert "Stage1" in result.output
        assert "Stage2" in result.output

    def test_list_models_shows_registry_keys(self) -> None:
        result = runner.invoke(app, ["--list-models"])
        assert "bs-roformer-vocals" in result.output
        assert "mel-roformer-karaoke" in result.output


class TestNoInput:
    def test_no_args_exits_one(self) -> None:
        result = runner.invoke(app, [])
        assert result.exit_code == 1

    def test_help_exits_zero(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0


class TestInvalidModel:
    def test_bad_stage2_model_exits_nonzero(self, tmp_path: Path, sample_wav: Path) -> None:
        """無効なモデル名を指定した場合、ゼロ以外の終了コードで失敗すること。"""
        with patch("chorus_extraction.config._check_cuda", return_value=False):
            result = runner.invoke(
                app,
                [str(sample_wav), "--stage2-model", "no-such-model", "--device", "cpu"],
            )
        assert result.exit_code != 0


class TestExtractVocal:
    def test_vocal_mode_calls_extract(self, tmp_path: Path, sample_wav: Path) -> None:
        from chorus_extraction.pipeline import ExtractionResult

        fake_result = ExtractionResult(
            input_path=sample_wav,
            lead_path=tmp_path / "sample_lead.wav",
            chorus_path=tmp_path / "sample_chorus.wav",
            all_output_paths=(
                tmp_path / "sample_lead.wav",
                tmp_path / "sample_chorus.wav",
            ),
        )

        with (
            patch("chorus_extraction.config._check_cuda", return_value=False),
            # make_separator と extract はモジュールレベルで cli に import 済み
            patch("chorus_extraction.cli.make_separator", return_value=object()),
            patch("chorus_extraction.cli.extract", return_value=[fake_result]),
        ):
            result = runner.invoke(
                app,
                [str(sample_wav), "--mode", "vocal", "--device", "cpu",
                 "--output-dir", str(tmp_path / "out")],
            )

        assert result.exit_code == 0
        assert sample_wav.name in result.output
