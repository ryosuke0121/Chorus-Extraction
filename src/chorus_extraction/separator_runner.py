"""audio-separator アダプタ。副作用（モデルDL・推論）をこのファイルに隔離する。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SeparationResult:
    """単一分離処理の結果。"""

    input_path: Path
    output_paths: tuple[Path, ...]
    """分離後に書き出されたファイルのパス一覧。"""


def make_separator(
    output_dir: Path,
    output_format: str,
    model_dir: Path,
    device: str,
    output_single_stem: str | None = None,
) -> object:
    """Separator インスタンスを生成して返す。

    use_autocast は CUDA 時のみ有効（CPU では AttributeError になるため False）。

    Returns:
        audio_separator.separator.Separator インスタンス。

    Raises:
        ImportError: audio-separator がインストールされていない場合。
        chorus_extraction.errors.ModelDownloadError: 初期化に失敗した場合。
    """
    from chorus_extraction.errors import ModelDownloadError

    try:
        from audio_separator.separator import Separator
    except ImportError as exc:
        raise ModelDownloadError(
            "audio-separator がインストールされていません。"
            " `pip install 'audio-separator[gpu]'` を実行してください。"
        ) from exc

    use_autocast = device == "cuda"

    try:
        sep: object = Separator(
            output_dir=str(output_dir.resolve()),
            output_format=output_format,
            model_file_dir=str(model_dir.resolve()),
            use_autocast=use_autocast,
            **({"output_single_stem": output_single_stem} if output_single_stem else {}),
        )
    except Exception as exc:
        raise ModelDownloadError(
            f"Separator の初期化に失敗しました: {exc}"
            f"\nモデルキャッシュディレクトリ: {model_dir}"
        ) from exc

    return sep


def run_separation(
    separator: object,
    model_filename: str,
    input_path: Path,
    output_names: dict[str, str] | None = None,
) -> SeparationResult:
    """モデルをロードし、1ファイルの分離処理を実行する。

    Args:
        separator: make_separator() で生成した Separator インスタンス。
        model_filename: ロードするモデルのファイル名（自動DL対象）。
        input_path: 分離する音声ファイルのパス。
        output_names: ステム名リネーム dict（例: {"Vocals": "track_lead"}）。

    Returns:
        SeparationResult: 出力ファイルパス一覧を持つ結果オブジェクト。

    Raises:
        chorus_extraction.errors.ModelDownloadError: モデルのロードに失敗した場合。
        chorus_extraction.errors.SeparationError: 分離処理中にエラーが発生した場合。
    """
    from chorus_extraction.errors import ModelDownloadError, SeparationError

    logger.info("モデルをロードしています: %s", model_filename)
    try:
        separator.load_model(model_filename=model_filename)  # type: ignore[attr-defined]
    except Exception as exc:
        raise ModelDownloadError(
            f"モデルのロードに失敗しました: {model_filename}"
            f"\n原因: {exc}"
            f"\nネットワーク接続とモデルキャッシュを確認してください。"
        ) from exc

    abs_input = input_path.resolve()
    logger.info("分離処理を開始します: %s", abs_input.name)
    try:
        raw_outputs: list[str] = separator.separate(  # type: ignore[attr-defined]
            str(abs_input),
            custom_output_names=output_names,
        )
    except Exception as exc:
        raise SeparationError(
            f"分離処理中にエラーが発生しました: {input_path.name}"
            f"\n原因: {exc}"
        ) from exc

    # audio-separator は相対パス（ファイル名のみ）を返す場合がある。
    # separator.output_dir を基準に絶対パスへ解決する。
    sep_output_dir = Path(getattr(separator, "output_dir", "."))
    resolved: list[Path] = []
    for p in raw_outputs:
        path = Path(p)
        if not path.is_absolute():
            path = (sep_output_dir / path).resolve()
        resolved.append(path)

    output_paths = tuple(resolved)
    logger.info("分離完了。出力ファイル数: %d", len(output_paths))
    return SeparationResult(input_path=input_path, output_paths=output_paths)
