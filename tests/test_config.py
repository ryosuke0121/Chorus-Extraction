"""config.py のユニットテスト。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from chorus_extraction.config import (
    DEFAULT_MULTI_STEM_MODEL,
    DEFAULT_STAGE1_MODEL,
    DEFAULT_STAGE2_MODEL,
    MODEL_REGISTRY,
    build_run_config,
    resolve_device,
)
from chorus_extraction.errors import DeviceUnavailableError, InvalidInputError


class TestModelRegistry:
    def test_default_models_exist(self) -> None:
        assert DEFAULT_STAGE1_MODEL in MODEL_REGISTRY
        assert DEFAULT_STAGE2_MODEL in MODEL_REGISTRY

    def test_stage_roles(self) -> None:
        for spec in MODEL_REGISTRY.values():
            assert spec.role in ("stage1", "stage2", "multi_stem")


class TestResolveDevice:
    def test_cpu_passthrough(self) -> None:
        assert resolve_device("cpu") == "cpu"

    def test_auto_uses_cuda_when_available(self) -> None:
        with patch("chorus_extraction.config._check_cuda", return_value=True):
            assert resolve_device("auto") == "cuda"

    def test_auto_falls_back_to_cpu(self) -> None:
        with patch("chorus_extraction.config._check_cuda", return_value=False):
            assert resolve_device("auto") == "cpu"

    def test_cuda_explicit_unavailable_raises(self) -> None:
        with patch("chorus_extraction.config._check_cuda", return_value=False), pytest.raises(DeviceUnavailableError):
            resolve_device("cuda")

    def test_cuda_explicit_available(self) -> None:
        with patch("chorus_extraction.config._check_cuda", return_value=True):
            assert resolve_device("cuda") == "cuda"


class TestBuildRunConfig:
    def test_defaults_full_mode(self, tmp_path: Path) -> None:
        """full/auto モードでは multi-stem モデルが stage1 の既定になる。"""
        with patch("chorus_extraction.config._check_cuda", return_value=False):
            cfg = build_run_config(
                inputs=[tmp_path / "a.wav"],
                mode="full",
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
        assert cfg.stage1_model.name == DEFAULT_MULTI_STEM_MODEL
        assert cfg.stage2_model.name == DEFAULT_STAGE2_MODEL
        assert cfg.device == "cpu"

    def test_defaults_song_mode(self, tmp_path: Path) -> None:
        """song モードでは mel-roformer-vocals が stage1 の既定になる。"""
        with patch("chorus_extraction.config._check_cuda", return_value=False):
            cfg = build_run_config(
                inputs=[tmp_path / "a.wav"],
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
        assert cfg.stage1_model.name == DEFAULT_STAGE1_MODEL
        assert cfg.stage2_model.name == DEFAULT_STAGE2_MODEL
        assert cfg.device == "cpu"

    def test_invalid_stage1_model_raises(self, tmp_path: Path) -> None:
        with pytest.raises(InvalidInputError, match="Stage1"):
            build_run_config(
                inputs=[tmp_path / "a.wav"],
                mode="song",
                output_dir=tmp_path,
                output_format="wav",
                stage1_model_name="nonexistent-model",
                stage2_model_name=None,
                model_dir=tmp_path,
                keep_intermediate=False,
                device="cpu",
                lead_name_template="{stem}_lead",
                chorus_name_template="{stem}_chorus",
                verbosity=0,
            )
