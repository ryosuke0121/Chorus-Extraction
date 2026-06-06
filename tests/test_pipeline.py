"""pipeline.py のユニットテスト。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from chorus_extraction.config import build_run_config
from chorus_extraction.errors import SeparationError
from chorus_extraction.pipeline import extract, separate_song, separate_vocal


def _make_cfg(tmp_path: Path, mode: str = "vocal", keep_intermediate: bool = False):
    with patch("chorus_extraction.config._check_cuda", return_value=False):
        return build_run_config(
            inputs=[tmp_path / "vocals.wav"],
            mode=mode,
            output_dir=tmp_path / "out",
            output_format="wav",
            stage1_model_name=None,
            stage2_model_name=None,
            model_dir=tmp_path / "models",
            keep_intermediate=keep_intermediate,
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


class TestSeparateSong:
    def test_identifies_lead_and_chorus(self, tmp_path: Path, mock_separator, make_separate_fn, sample_wav: Path) -> None:
        """Stage1 で Vocals を特定し、Stage2 で lead/chorus を正しく識別する。"""
        separate_fn = make_separate_fn(
            ["song_Vocals.wav", "song_Instrumental.wav"],
            ["song_lead.wav", "song_chorus.wav"],
        )
        cfg = _make_cfg(tmp_path, "song")

        result = separate_song(sample_wav, cfg, separator=mock_separator, separate_fn=separate_fn)

        assert result.input_path == sample_wav
        assert result.lead_path is not None
        assert result.lead_path.name == "song_lead.wav"
        assert result.chorus_path is not None
        assert result.chorus_path.name == "song_chorus.wav"

    def test_stage1_vocals_missing_raises(self, tmp_path: Path, mock_separator, make_separate_fn, sample_wav: Path) -> None:
        """Stage1 の出力に Vocals ステムがない場合は SeparationError を送出する。"""
        separate_fn = make_separate_fn(
            ["song_Accompaniment.wav", "song_Drums.wav"],
        )
        cfg = _make_cfg(tmp_path, "song")

        with pytest.raises(SeparationError, match="Vocals ステムが見つかりません"):
            separate_song(sample_wav, cfg, separator=mock_separator, separate_fn=separate_fn)

    def test_keep_intermediate_preserves_stage1(self, tmp_path: Path, mock_separator, make_separate_fn, sample_wav: Path) -> None:
        """keep_intermediate=True のとき Stage1 ファイルが output/intermediate/ に残る。"""
        separate_fn = make_separate_fn(
            ["song_Vocals.wav", "song_Instrumental.wav"],
            ["song_lead.wav", "song_chorus.wav"],
        )
        cfg = _make_cfg(tmp_path, "song", keep_intermediate=True)

        separate_song(sample_wav, cfg, separator=mock_separator, separate_fn=separate_fn)

        intermediate = cfg.output_dir / "intermediate"
        assert (intermediate / "song_Vocals.wav").exists()
        assert (intermediate / "song_Instrumental.wav").exists()

    def test_without_keep_intermediate_stage1_cleaned_up(self, tmp_path: Path, mock_separator, make_separate_fn, sample_wav: Path) -> None:
        """keep_intermediate=False のとき Stage1 ファイルは元の場所から削除される。"""
        stage1_files = ["song_Vocals.wav", "song_Instrumental.wav"]
        separate_fn = make_separate_fn(
            stage1_files,
            ["song_lead.wav", "song_chorus.wav"],
        )
        cfg = _make_cfg(tmp_path, "song", keep_intermediate=False)

        separate_song(sample_wav, cfg, separator=mock_separator, separate_fn=separate_fn)

        for name in stage1_files:
            assert not (tmp_path / name).exists()


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

    def test_song_mode(self, tmp_path: Path, mock_separator, make_separate_fn, sample_wav: Path) -> None:
        """song モードで 2 段階処理が実行される。"""
        separate_fn = make_separate_fn(
            ["sample_Vocals.wav", "sample_Instrumental.wav"],
            ["sample_lead.wav", "sample_chorus.wav"],
        )
        with patch("chorus_extraction.config._check_cuda", return_value=False):
            cfg = build_run_config(
                inputs=[sample_wav],
                mode="song",
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
        assert results[0].lead_path is not None
        assert results[0].chorus_path is not None

    def test_auto_mode_uses_song_pipeline(self, tmp_path: Path, mock_separator, make_separate_fn, sample_wav: Path) -> None:
        """mode='auto' は 2 段階の song パイプラインとして動作する。"""
        separate_fn = make_separate_fn(
            ["sample_Vocals.wav", "sample_Instrumental.wav"],
            ["sample_lead.wav", "sample_chorus.wav"],
        )
        with patch("chorus_extraction.config._check_cuda", return_value=False):
            cfg = build_run_config(
                inputs=[sample_wav],
                mode="auto",
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
        assert results[0].lead_path is not None

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
