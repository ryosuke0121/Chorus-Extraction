# chorus-extract

楽曲・ボーカル音源からコーラス（ハモリ／バッキングボーカル）を抽出する CLI ツール。  
[audio-separator](https://github.com/nomadkaraoke/python-audio-separator)（MIT）が提供する UVR モデル群を活用した 2 段階パイプラインで、リードボーカルとハモリを分離します。

## 特徴

- **2 段階処理 (`--mode song`)**: ミックス済み楽曲 → ボーカル抽出 (Stage1) → リード / ハモリ分離 (Stage2)
- **1 段階処理 (`--mode vocal`)**: ボーカル音源を直接リード / ハモリ分離
- NVIDIA GPU (CUDA) による高速推論
- モデルの自動ダウンロード・キャッシュ
- 複数ファイルのバッチ処理

## 動作環境

| 項目 | 要件 |
|------|------|
| Python | 3.12 |
| GPU | NVIDIA GPU + CUDA 12.8 推奨（CPU 動作も可） |
| OS | Windows / Linux / macOS |

## インストール

```bash
# リポジトリのクローン
git clone https://github.com/ryosuke0121/Chorus-Extraction.git
cd Chorus-Extraction

# 仮想環境の作成と依存関係のインストール
uv venv --python 3.12
uv sync
```

> **GPU 使用時**: `pyproject.toml` に PyTorch CUDA インデックスが設定済みです。  
> `uv sync` 実行時に自動で CUDA ビルドの torch がインストールされます。

## 使い方

```bash
# ミックス済み楽曲からコーラスを抽出（2 段階）
chorus-extract --mode song song.mp3

# ボーカル音源から直接分離（1 段階）
chorus-extract --mode vocal vocal.wav

# 複数ファイル・オプション指定
chorus-extract --mode song --output-dir ./out --keep-intermediate -v -- song1.mp3 song2.flac
```

### 主要オプション

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--mode` | `auto` | `song`=完成曲 (2段階), `vocal`=ボーカル音源 (1段階), `auto`=自動 |
| `--output-dir` / `-o` | `output` | 出力ディレクトリ |
| `--output-format` | `wav` | 出力形式 (`wav` / `mp3` / `flac` / `m4a` / `ogg`) |
| `--stage1-model` | `bs-roformer-vocals` | Stage1 モデル名 |
| `--stage2-model` | `mel-roformer-karaoke` | Stage2 モデル名 |
| `--model-dir` | `models` | モデルキャッシュディレクトリ |
| `--keep-intermediate` | OFF | Stage1 の中間ステムを保持する |
| `--device` | `auto` | `auto` / `cuda` / `cpu` |
| `--lead-name` | `{stem}_lead` | リードボーカル出力名テンプレート |
| `--chorus-name` | `{stem}_chorus` | ハモリ出力名テンプレート |
| `-v` / `-vv` | — | ログ詳細度 (INFO / DEBUG) |
| `--list-models` | — | 利用可能なモデル一覧を表示 |

> **引数の順序**: ファイルパスに特殊文字や空白が含まれる場合は `--` の後に指定してください。

## 採用モデル

```
chorus-extract --list-models
```

| キー | 用途 | 説明 |
|------|------|------|
| `bs-roformer-vocals` | Stage1（既定） | BS-Roformer ボーカル分離 (SDR 12.97) |
| `mel-roformer-karaoke` | Stage2（既定） | Mel-Band Roformer カラオケ aufr33/viperx (SDR 10.20) |
| `uvr-bve` | Stage2 | UVR BVE バッキングボーカル抽出 |
| `uvr-hp-karaoke` | Stage2 | UVR HP-Karaoke 6 |
| `uvr-mdx-karaoke` | Stage2 | UVR MDX-Net Karaoke 2 |

## 出力ファイル

```
output/
├── {stem}_lead.wav      # リードボーカル
└── {stem}_chorus.wav    # ハモリ（バッキングボーカル）

# --keep-intermediate 指定時
output/intermediate/
├── {stem}_(Vocals)_*.wav        # Stage1: ボーカル抽出結果
└── {stem}_(Instrumental)_*.wav  # Stage1: 伴奏抽出結果
```

## 開発

```bash
# テスト実行
uv run pytest

# 型チェック
uv run mypy src

# リント
uv run ruff check .

# 統合テスト（実モデルDL必要）
uv run pytest -m integration
```

## ライセンス

MIT License — モデル重みは各プロジェクトのライセンスに従います。商用利用時は各モデルの配布元ライセンスを確認してください。
