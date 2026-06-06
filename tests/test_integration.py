"""統合テスト。実モデルのダウンロードと推論を伴う。

pytest -m integration で実行（既定スキップ）。
NVIDIA GPU + audio-separator[gpu] インストール済み環境が必要。
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture()
def short_wav(tmp_path: Path) -> Path:
    """1秒分のサイレント WAV ファイルを生成する。"""
    import struct
    import wave

    path = tmp_path / "silence_1s.wav"
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(struct.pack("<" + "h" * (44100 * 2), *([0] * (44100 * 2))))
    return path


def test_vocal_mode_real_model(tmp_path: Path, short_wav: Path) -> None:
    """vocal モードで Mel-Band Roformer karaoke モデルを実際に実行する。"""
    from chorus_extraction.config import build_run_config
    from chorus_extraction.pipeline import extract
    from chorus_extraction.separator_runner import make_separator, run_separation

    cfg = build_run_config(
        inputs=[short_wav],
        mode="vocal",
        output_dir=tmp_path / "out",
        output_format="wav",
        stage1_model_name=None,
        stage2_model_name=None,
        model_dir=tmp_path / "models",
        keep_intermediate=False,
        device="auto",
        lead_name_template="{stem}_lead",
        chorus_name_template="{stem}_chorus",
        verbosity=1,
    )

    separator = make_separator(
        output_dir=cfg.output_dir,
        output_format=cfg.output_format,
        model_dir=cfg.model_dir,
        device=cfg.device,
    )
    results = extract(cfg, separator=separator, separate_fn=run_separation)

    assert len(results) == 1
    assert results[0].lead_path is not None or results[0].chorus_path is not None
