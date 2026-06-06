"""設定 dataclass とモデルレジストリ。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# 型エイリアス
# ---------------------------------------------------------------------------
Mode = Literal["auto", "song", "vocal"]
Device = Literal["auto", "cuda", "cpu"]
OutputFormat = Literal["wav", "mp3", "flac", "m4a", "ogg"]


# ---------------------------------------------------------------------------
# モデル仕様
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    """単一モデルの仕様。"""

    name: str
    filename: str
    arch: str
    role: Literal["stage1", "stage2"]
    description: str


# ---------------------------------------------------------------------------
# モデルレジストリ
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, ModelSpec] = {
    # Stage 1: ミックス → ボーカル抽出
    "bs-roformer-vocals": ModelSpec(
        name="bs-roformer-vocals",
        filename="model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        arch="bs_roformer",
        role="stage1",
        description="BS-Roformer ボーカル分離モデル（SDR 12.97、Stage1 既定）",
    ),
    # Stage 2: ボーカル → リード / ハモリ分離
    "mel-roformer-karaoke": ModelSpec(
        name="mel-roformer-karaoke",
        filename="mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
        arch="mel_band_roformer",
        role="stage2",
        description="Mel-Band Roformer カラオケモデル aufr33/viperx（SDR 10.20、Stage2 既定）",
    ),
    "uvr-bve": ModelSpec(
        name="uvr-bve",
        filename="UVR-BVE-4B_SN-44100-1.pth",
        arch="vr_arch",
        role="stage2",
        description="UVR BVE バッキングボーカル抽出（VR Architecture）",
    ),
    "uvr-hp-karaoke": ModelSpec(
        name="uvr-hp-karaoke",
        filename="6_HP-Karaoke-UVR.pth",
        arch="vr_arch",
        role="stage2",
        description="UVR HP-Karaoke 6（VR Architecture）",
    ),
    "uvr-mdx-karaoke": ModelSpec(
        name="uvr-mdx-karaoke",
        filename="UVR_MDXNET_KARA_2.onnx",
        arch="mdx_net",
        role="stage2",
        description="UVR MDX-Net Karaoke 2（ONNX）",
    ),
}

DEFAULT_STAGE1_MODEL = "bs-roformer-vocals"
DEFAULT_STAGE2_MODEL = "mel-roformer-karaoke"

SUPPORTED_INPUT_EXTENSIONS: frozenset[str] = frozenset(
    {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aiff", ".aif", ".wma"}
)


# ---------------------------------------------------------------------------
# 実行設定
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RunConfig:
    """CLI から解析された実行設定。"""

    inputs: tuple[Path, ...]
    mode: Mode
    output_dir: Path
    output_format: OutputFormat
    stage1_model: ModelSpec
    stage2_model: ModelSpec
    model_dir: Path
    keep_intermediate: bool
    device: Device
    lead_name_template: str
    chorus_name_template: str
    verbosity: int


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------
def resolve_device(requested: Device) -> Device:
    """要求されたデバイスを検証し、利用可能な実デバイスを返す。

    "auto" の場合は CUDA の利用可否を検出し、不可なら警告して "cpu" を返す。
    "cuda" を明示して利用不可なら DeviceUnavailableError を送出。
    """
    import logging

    from chorus_extraction.errors import DeviceUnavailableError

    logger = logging.getLogger(__name__)

    if requested == "cpu":
        return "cpu"

    cuda_available = _check_cuda()

    if requested == "cuda":
        if not cuda_available:
            raise DeviceUnavailableError(
                "CUDA が利用できません。"
                " NVIDIA GPU ドライバおよび CUDA Toolkit がインストールされているか確認してください。"
                " CPU で実行する場合は --device cpu を指定してください。"
            )
        return "cuda"

    # auto
    if cuda_available:
        logger.info("CUDA を検出しました。GPU で処理します。")
        return "cuda"

    logger.warning(
        "CUDA が利用できないため CPU で処理します。処理に時間がかかる場合があります。"
        " NVIDIA GPU 環境では `pip install audio-separator[gpu]` を実行してください。"
    )
    return "cpu"


def _check_cuda() -> bool:
    """CUDA が利用可能かどうかを判定する。"""
    import logging

    _logger = logging.getLogger(__name__)

    try:
        import torch  # type: ignore[import]

        return bool(torch.cuda.is_available())
    except ImportError:
        pass
    except Exception as exc:
        _logger.debug("torch による CUDA 検出に失敗しました: %s", exc)

    try:
        import onnxruntime  # type: ignore[import]

        return "CUDAExecutionProvider" in onnxruntime.get_available_providers()
    except ImportError:
        pass
    except Exception as exc:
        _logger.debug("onnxruntime による CUDA 検出に失敗しました: %s", exc)

    return False


_VALID_MODES: frozenset[str] = frozenset({"auto", "song", "vocal"})
_VALID_DEVICES: frozenset[str] = frozenset({"auto", "cuda", "cpu"})
_VALID_FORMATS: frozenset[str] = frozenset({"wav", "mp3", "flac", "m4a", "ogg"})


def build_run_config(
    inputs: list[Path],
    mode: str,
    output_dir: Path,
    output_format: str,
    stage1_model_name: str | None,
    stage2_model_name: str | None,
    model_dir: Path,
    keep_intermediate: bool,
    device: str,
    lead_name_template: str,
    chorus_name_template: str,
    verbosity: int,
) -> RunConfig:
    """CLI 引数から RunConfig を構築する。モデル名の解決と検証を担う。"""
    from chorus_extraction.errors import InvalidInputError

    # mode / device / output_format のランタイム検証
    if mode not in _VALID_MODES:
        raise InvalidInputError(
            f"無効な mode です: '{mode}'。指定可能: {', '.join(sorted(_VALID_MODES))}"
        )
    if device not in _VALID_DEVICES:
        raise InvalidInputError(
            f"無効な device です: '{device}'。指定可能: {', '.join(sorted(_VALID_DEVICES))}"
        )
    if output_format not in _VALID_FORMATS:
        raise InvalidInputError(
            f"非対応の出力フォーマットです: '{output_format}'。"
            f" 対応フォーマット: {', '.join(sorted(_VALID_FORMATS))}"
        )

    # モデル解決
    s1_key = stage1_model_name or DEFAULT_STAGE1_MODEL
    s2_key = stage2_model_name or DEFAULT_STAGE2_MODEL

    if s1_key not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise InvalidInputError(
            f"Stage1 モデル '{s1_key}' はレジストリに存在しません。"
            f" 利用可能: {available}"
        )
    if s2_key not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise InvalidInputError(
            f"Stage2 モデル '{s2_key}' はレジストリに存在しません。"
            f" 利用可能: {available}"
        )

    # ロール検証（stage1 モデルを stage2 に使う誤指定を早期検出）
    if MODEL_REGISTRY[s1_key].role != "stage1":
        raise InvalidInputError(
            f"'{s1_key}' は Stage1 モデルではありません"
            f" (role={MODEL_REGISTRY[s1_key].role})。"
        )
    if MODEL_REGISTRY[s2_key].role != "stage2":
        raise InvalidInputError(
            f"'{s2_key}' は Stage2 モデルではありません"
            f" (role={MODEL_REGISTRY[s2_key].role})。"
        )

    return RunConfig(
        inputs=tuple(inputs),
        mode=mode,  # type: ignore[arg-type]
        output_dir=output_dir,
        output_format=output_format,  # type: ignore[arg-type]
        stage1_model=MODEL_REGISTRY[s1_key],
        stage2_model=MODEL_REGISTRY[s2_key],
        model_dir=model_dir,
        keep_intermediate=keep_intermediate,
        device=resolve_device(device),  # type: ignore[arg-type]
        lead_name_template=lead_name_template,
        chorus_name_template=chorus_name_template,
        verbosity=verbosity,
    )
