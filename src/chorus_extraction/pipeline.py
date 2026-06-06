"""純粋オーケストレーション。分離の副作用は separator_runner 経由で注入する（DI）。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from chorus_extraction.audio_io import build_output_names, intermediate_dir, validate_input
from chorus_extraction.config import RunConfig
from chorus_extraction.separator_runner import SeparationResult

logger = logging.getLogger(__name__)

# 分離関数の型エイリアス（DI / テスト用モック差替に使用）
SeparateFn = Callable[
    [object, str, Path, "dict[str, str] | None"],
    SeparationResult,
]


@dataclass(frozen=True)
class ExtractionResult:
    """1入力ファイルに対するコーラス抽出の最終結果。"""

    input_path: Path
    lead_path: Path | None
    """リードボーカルの出力パス（取得できた場合）。"""
    chorus_path: Path | None
    """ハモリ（バッキングボーカル）の出力パス（取得できた場合）。"""
    all_output_paths: tuple[Path, ...]
    """分離で生成されたすべてのファイルパス。"""


def separate_vocal(
    input_path: Path,
    cfg: RunConfig,
    *,
    separator: object,
    separate_fn: SeparateFn,
) -> ExtractionResult:
    """ボーカル音源（伴奏なし）からリード / ハモリを1段階で分離する。"""
    import tempfile

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    output_names = build_output_names(stem, cfg)

    with tempfile.TemporaryDirectory(prefix="chorus_extraction_") as _tmp:
        safe_input = _ensure_safe_input(input_path, Path(_tmp))
        result = separate_fn(
            separator,
            cfg.stage2_model.filename,
            safe_input,
            output_names,
        )

    return _build_extraction_result(input_path, result, output_names)


def separate_song(
    input_path: Path,
    cfg: RunConfig,
    *,
    separator: object,
    separate_fn: SeparateFn,
) -> ExtractionResult:
    """完成曲（ミックス済み）から2段階でハモリを抽出する。

    Stage1: ボーカル / 伴奏分離
    Stage2: リード / ハモリ分離
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    with intermediate_dir(cfg) as tmp_dir:
        # 非ASCIIパスはlibsoundfileが開けないため安全名でコピーする
        safe_input = _ensure_safe_input(input_path, tmp_dir)

        # Stage 1: ミックス → ボーカル抽出
        logger.info("[Stage1] ボーカル / 伴奏分離を開始: %s", input_path.name)

        stage1_result = separate_fn(
            separator,
            cfg.stage1_model.filename,
            safe_input,
            None,
        )

        # Vocals ステムを探す
        vocal_path = _find_stem_path(stage1_result.output_paths, "Vocals")
        if vocal_path is None:
            raise _make_stage1_error(stage1_result.output_paths)

        # Stage1 中間ファイルを tmp_dir に移動（keep_intermediate=True なら intermediate/、False なら自動削除 tmp）
        _move_to_dir(stage1_result.output_paths, tmp_dir)
        vocal_path = tmp_dir / vocal_path.name

        # Stage 2: ボーカル → リード / ハモリ
        logger.info("[Stage2] リード / ハモリ分離を開始: %s", vocal_path.name)
        stem = input_path.stem
        output_names = build_output_names(stem, cfg)

        stage2_result = separate_fn(
            separator,
            cfg.stage2_model.filename,
            vocal_path,
            output_names,
        )

    return _build_extraction_result(input_path, stage2_result, output_names)


def extract(
    cfg: RunConfig,
    *,
    separator: object,
    separate_fn: SeparateFn,
) -> list[ExtractionResult]:
    """RunConfig に含まれる全入力ファイルを処理して結果リストを返す。

    mode が "auto" の場合は "song"（2段階）として扱う。
    """
    results: list[ExtractionResult] = []

    for input_path in cfg.inputs:
        validate_input(input_path)
        effective_mode = cfg.mode if cfg.mode != "auto" else "song"

        if effective_mode == "vocal":
            result = separate_vocal(
                input_path, cfg, separator=separator, separate_fn=separate_fn
            )
        else:
            result = separate_song(
                input_path, cfg, separator=separator, separate_fn=separate_fn
            )

        results.append(result)

    return results


# ---------------------------------------------------------------------------
# 内部ユーティリティ
# ---------------------------------------------------------------------------
def _find_stem_path(paths: tuple[Path, ...], stem_keyword: str) -> Path | None:
    """ファイル名に stem_keyword を含むパスを返す（大文字小文字無視）。"""
    keyword_lower = stem_keyword.lower()
    for p in paths:
        if keyword_lower in p.name.lower():
            return p
    return None


def _build_extraction_result(
    input_path: Path,
    result: SeparationResult,
    output_names: dict[str, str],
) -> ExtractionResult:
    """SeparationResult と output_names からリード / ハモリのパスを特定して返す。"""
    lead_name = output_names.get("Vocals", "")
    chorus_name = output_names.get("Instrumental", "")

    lead_path: Path | None = None
    chorus_path: Path | None = None

    for p in result.output_paths:
        name_no_ext = p.stem
        if lead_name and name_no_ext == lead_name:
            lead_path = p
        elif chorus_name and name_no_ext == chorus_name:
            chorus_path = p

    # テンプレートによる完全一致がなければキーワードで探す（モデルごとにステム名が異なるため複数候補を試行）
    if lead_path is None:
        for kw in ("lead", "vocals", "vocal"):
            lead_path = _find_stem_path(result.output_paths, kw)
            if lead_path is not None:
                break
    # ハモリは lead に使われなかった残りのファイルから特定する
    if chorus_path is None:
        for kw in ("chorus", "backing", "instrumental", "bv", "harmony"):
            candidate = _find_stem_path(result.output_paths, kw)
            if candidate is not None and candidate != lead_path:
                chorus_path = candidate
                break
    # それでも見つからなければ lead_path 以外の最初のファイルを充てる
    if chorus_path is None:
        for p in result.output_paths:
            if p != lead_path:
                chorus_path = p
                break

    return ExtractionResult(
        input_path=input_path,
        lead_path=lead_path,
        chorus_path=chorus_path,
        all_output_paths=result.output_paths,
    )


def _make_stage1_error(output_paths: tuple[Path, ...]) -> Exception:
    from chorus_extraction.errors import SeparationError

    files = ", ".join(p.name for p in output_paths)
    return SeparationError(
        f"Stage1 の出力から Vocals ステムが見つかりません。"
        f"\n生成されたファイル: {files}"
        f"\n使用するモデルが Vocals ステムを出力するか確認してください。"
    )


def _ensure_safe_input(input_path: Path, tmp_dir: Path) -> Path:
    """パスにシステムロケール非対応文字が含まれる場合、ASCII-safe名でコピーして返す。

    libsoundfile は ANSI API を使う場合があり、CP932 非対応文字を含むパスを開けない。
    """
    import shutil

    try:
        str(input_path).encode("cp932")
        return input_path  # 変換可能ならそのまま使う
    except (UnicodeEncodeError, LookupError):
        safe_path = tmp_dir / f"input{input_path.suffix}"
        shutil.copy2(input_path, safe_path)
        logger.info("非ASCII入力を一時コピー: %s -> %s", input_path.name, safe_path.name)
        return safe_path


def _move_to_dir(paths: tuple[Path, ...], dest: Path) -> None:
    """ファイル群を dest ディレクトリに移動する。同名ファイルが存在する場合は警告する。"""
    import shutil

    for p in paths:
        if not p.exists():
            continue
        dst = dest / p.name
        if dst.exists():
            logger.warning(
                "移動先に同名ファイルが既に存在するため上書きします: %s", dst
            )
        shutil.move(str(p), dst)
