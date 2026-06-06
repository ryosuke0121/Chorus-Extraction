"""Typer ベースの CLI シェル。引数解析 → RunConfig 構築 → extract 実行 → 結果表示。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from chorus_extraction.config import (
    MODEL_REGISTRY,
    RunConfig,
    build_run_config,
)
from chorus_extraction.errors import ChorusExtractionError, InvalidInputError
from chorus_extraction.logging_setup import configure_logging
from chorus_extraction.pipeline import ExtractionResult, extract
from chorus_extraction.separator_runner import make_separator, run_separation

app = typer.Typer(
    name="chorus-extract",
    help="楽曲・ボーカル音源からコーラス（ハモリ・バッキングボーカル）を抽出するツール。",
    add_completion=False,
)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    inputs: Annotated[
        list[Path] | None,
        typer.Argument(help="入力音声ファイル（複数指定可）"),
    ] = None,
    mode: Annotated[
        str,
        typer.Option(
            "--mode",
            "-m",
            help="入力種別: full=全ステム分離(既定), song=ボーカル2段階, vocal=ボーカル音源1段階, auto=fullの別名",
        ),
    ] = "full",
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="出力ディレクトリ"),
    ] = Path("output"),
    output_format: Annotated[
        str,
        typer.Option("--output-format", help="出力フォーマット（wav/mp3/flac/m4a/ogg）"),
    ] = "wav",
    stage1_model: Annotated[
        str | None,
        typer.Option("--stage1-model", help="Stage1 モデル名（--list-models で確認）"),
    ] = None,
    stage2_model: Annotated[
        str | None,
        typer.Option("--stage2-model", help="Stage2 モデル名（--list-models で確認）"),
    ] = None,
    model_dir: Annotated[
        Path,
        typer.Option("--model-dir", help="モデル重みのキャッシュディレクトリ"),
    ] = Path("models"),
    keep_intermediate: Annotated[
        bool,
        typer.Option("--keep-intermediate", help="2段階処理の中間ステムファイルを保持する"),
    ] = False,
    device: Annotated[
        str,
        typer.Option("--device", help="実行デバイス: auto=自動検出, cuda=GPU強制, cpu=CPU強制"),
    ] = "auto",
    lead_name_template: Annotated[
        str,
        typer.Option("--lead-name", help="リードボーカル出力ファイル名テンプレート（{stem} を使用可）"),
    ] = "{stem}_lead",
    chorus_name_template: Annotated[
        str,
        typer.Option("--chorus-name", help="ハモリ出力ファイル名テンプレート（{stem} を使用可）"),
    ] = "{stem}_chorus",
    list_models: Annotated[
        bool,
        typer.Option("--list-models", help="利用可能なモデル一覧を表示して終了"),
    ] = False,
    verbose: Annotated[
        int,
        typer.Option("--verbose", "-v", count=True, help="ログ詳細度（-v で INFO、-vv で DEBUG）"),
    ] = 0,
) -> None:
    """コーラス抽出を実行する。"""
    configure_logging(verbose)

    # --list-models
    if list_models:
        _print_model_list()
        raise typer.Exit(0)

    # 入力ファイルチェック
    if not inputs:
        typer.echo("エラー: 入力ファイルを指定してください。", err=True)
        typer.echo("使用方法: chorus-extract INPUT... [OPTIONS]", err=True)
        typer.echo("ヘルプ:   chorus-extract --help", err=True)
        raise typer.Exit(1)

    try:
        cfg = build_run_config(
            inputs=list(inputs),
            mode=mode,
            output_dir=output_dir,
            output_format=output_format,
            stage1_model_name=stage1_model,
            stage2_model_name=stage2_model,
            model_dir=model_dir,
            keep_intermediate=keep_intermediate,
            device=device,
            lead_name_template=lead_name_template,
            chorus_name_template=chorus_name_template,
            verbosity=verbose,
        )
    except InvalidInputError as exc:
        typer.echo(f"設定エラー: {exc}", err=True)
        raise typer.Exit(1) from exc
    except ChorusExtractionError as exc:
        typer.echo(f"エラー: {exc}", err=True)
        raise typer.Exit(1) from exc

    _run_extraction(cfg)


def _run_extraction(cfg: RunConfig) -> None:
    """分離処理を実行し結果を表示する。"""
    try:
        separator = make_separator(
            output_dir=cfg.output_dir,
            output_format=cfg.output_format,
            model_dir=cfg.model_dir,
            device=cfg.device,
        )
        results = extract(cfg, separator=separator, separate_fn=run_separation)
    except ChorusExtractionError as exc:
        typer.echo(f"処理エラー: {exc}", err=True)
        raise typer.Exit(2) from exc
    except Exception as exc:
        typer.echo(f"予期しないエラーが発生しました: {exc}", err=True)
        if cfg.verbosity >= 1:
            import traceback
            traceback.print_exc(file=sys.stderr)
        raise typer.Exit(2) from exc

    for result in results:
        typer.echo(f"\n[OK] {result.input_path.name}")
        if result.lead_path:
            typer.echo(f"  リード      : {result.lead_path}")
        if result.chorus_path:
            typer.echo(f"  ハモリ      : {result.chorus_path}")
        _print_stems(result)


_STEM_LABEL_JP: dict[str, str] = {
    "guitar": "ギター",
    "bass": "ベース",
    "drums": "ドラム",
    "keyboard": "キーボード",
    "other": "その他",
}


def _print_stems(result: ExtractionResult) -> None:
    """ExtractionResult の stems（追加ステム）を表示する。stems がない場合は all_output_paths の残りを表示。"""
    known = {result.lead_path, result.chorus_path}
    if result.stems:
        for label, path in result.stems.items():
            jp = _STEM_LABEL_JP.get(label, label)
            typer.echo(f"  {jp:<10}: {path}")
    else:
        for p in result.all_output_paths:
            if p not in known:
                typer.echo(f"  その他      : {p}")


def _print_model_list() -> None:
    """モデルレジストリの一覧をフォーマットして表示する。"""
    multi_models = {k: v for k, v in MODEL_REGISTRY.items() if v.role == "multi_stem"}
    stage1_models = {k: v for k, v in MODEL_REGISTRY.items() if v.role == "stage1"}
    stage2_models = {k: v for k, v in MODEL_REGISTRY.items() if v.role == "stage2"}

    typer.echo("\n=== Multi-Stem モデル（full mode: ミックス → 全ステム同時分離）===")
    for key, spec in multi_models.items():
        typer.echo(f"  {key:<30} {spec.description}")

    typer.echo("\n=== Stage1 モデル（song mode: ミックス → ボーカル抽出）===")
    for key, spec in stage1_models.items():
        typer.echo(f"  {key:<30} {spec.description}")

    typer.echo("\n=== Stage2 モデル（ボーカル → リード / ハモリ分離）===")
    for key, spec in stage2_models.items():
        typer.echo(f"  {key:<30} {spec.description}")

    typer.echo("\n使用例:")
    typer.echo("  chorus-extract song.wav                          # full mode（既定）")
    typer.echo("  chorus-extract song.wav --mode song              # song mode（ボーカル2段階）")
    typer.echo("  chorus-extract song.wav --stage2-model uvr-bve   # Stage2 モデル変更")
