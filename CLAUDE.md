# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# 依存関係インストール
uv sync

# 実行（full mode が既定）
uv run chorus-extract -- input.mp3

# song mode（ボーカル2段階のみ）
uv run chorus-extract --mode song -- input.mp3

# テスト（統合テストはデフォルトスキップ）
uv run pytest
uv run pytest tests/test_pipeline.py::test_foo   # 単一テスト
uv run pytest -m integration                      # 実モデル使用統合テスト

# 型チェック・リント
uv run mypy src
uv run ruff check .
uv run ruff format .
```

## Architecture

**3 モード**:
- `full`（既定）: htdemucs_6s で 6 ステム → karaoke モデルで Vocals をリード/ハモリ分離 → 7 ファイル出力
- `song`: mel-roformer-vocals でボーカル抽出（Stage1）→ karaoke モデルでリード/ハモリ分離（Stage2）
- `vocal`: Stage2 のみ（ボーカル音源入力）

### モジュール依存関係

```
cli.py → config.py (RunConfig 構築)
       → pipeline.py (extract / separate_full / separate_song / separate_vocal)
       → separator_runner.py (make_separator / run_separation)

pipeline.py → audio_io.py (validate_input, build_output_names, intermediate_dir)
            → separator_runner.py (SeparationResult 型のみ参照)
```

### 設計方針

- **副作用の境界**: `separator_runner.py` に `audio-separator` ライブラリ呼び出しを隔離。`pipeline.py` は純粋オーケストレーション。
- **DI によるテスト容易性**: `extract()` / `separate_full()` / `separate_song()` / `separate_vocal()` はすべて `separator` と `separate_fn` を引数で受け取る。テストでは `conftest.py` の `make_separate_fn` スタブを使う。
- **frozen dataclass**: `RunConfig` / `ModelSpec` / `SeparationResult` / `ExtractionResult` はすべてイミュータブル（`ExtractionResult.stems` は `dict` だが field reference は frozen）。

### Windows 固有の注意事項

- **CP932 非対応文字**: `⧸` (U+29F8) などを含むファイル名は libsoundfile が開けない。`pipeline._ensure_safe_input()` が ASCII-safe な一時パスにコピーしてから分離処理に渡す。
- **audio-separator の相対パス返却**: `separator.separate()` はファイル名のみ（相対パス）を返す。`separator_runner.run_separation()` で `separator.output_dir` を基準に絶対パスへ解決済み。
- **CLI の `--` 区切り**: Typer のバリアディック引数では、特殊文字を含むファイルパスは `--` の後に指定する（例: `chorus-extract -- "file⧸name.mp3"`）。

### CUDA / PyTorch

`torch` を直接依存に追加し `[tool.uv.sources]` で cu128 インデックスを指定している（トランジティブ依存には sources が効かないため）。`uv.lock` を更新する際は `uv lock --upgrade-package torch && uv sync` を使う。

### モデルレジストリ

`config.MODEL_REGISTRY` に全モデル定義。ロールは `stage1` / `stage2` / `multi_stem` の 3 種。`build_run_config()` がモードとロールの不一致を早期検出する。

| ロール | 用途 | デフォルト |
|--------|------|-----------|
| `multi_stem` | full/auto モードの Stage1（全ステム分離） | `htdemucs-6s` |
| `stage1` | song モードの Stage1（ボーカル抽出） | `mel-roformer-vocals` |
| `stage2` | 全モードの Stage2（リード/ハモリ分離） | `mel-roformer-karaoke` |

カラオケ系モデルは `Vocals`=リード / `Instrumental`=ハモリ の 2 ステムを出力する。  
htdemucs_6s は `vocals / drums / bass / guitar / piano / other` の 6 ステムを出力し、`piano` は `keyboard` にリネームされる。

### 終了コード

| コード | 意味 |
|--------|------|
| 0 | 成功 |
| 1 | ユーザー起因エラー（InvalidInput / DeviceUnavailable） |
| 2 | 実行時エラー（ModelDownload / Separation） |
