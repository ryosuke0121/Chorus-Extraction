# chorus-extract

楽曲から **ギター・ベース・ドラム・キーボード・その他・リードボーカル・コーラス** を一括分離する CLI ツール。  
[audio-separator](https://github.com/nomadkaraoke/python-audio-separator)（MIT）が提供するモデル群を活用したマルチステム分離パイプラインです。

## 特徴

- **Full mode（既定）**: ミックス済み楽曲 → 7 ステム同時分離（ギター / ベース / ドラム / キーボード / その他 / リード / ハモリ）
- **Song mode**: 2 段階でリード / ハモリのみ分離（高速）
- **Vocal mode**: ボーカル音源を直接リード / ハモリ分離（1 段階）
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

[uv](https://docs.astral.sh/uv/) が必要です（未インストールの場合は公式サイトの手順に従ってください）。

```bash
uv tool install git+https://github.com/ryosuke0121/Chorus-Extraction.git
```

インストール後は `chorus-extract` コマンドがそのまま使えます。

> **GPU 使用時**: Windows / Linux x86_64 では CUDA 12.8 対応の PyTorch が自動でインストールされます。

### 開発者向け

```bash
git clone https://github.com/ryosuke0121/Chorus-Extraction.git
cd Chorus-Extraction
uv sync
uv run chorus-extract --help
```

## 使い方

```bash
# 楽曲から全ステム分離（既定: full mode）
chorus-extract song.mp3

# リード / ハモリのみ（song mode）
chorus-extract --mode song song.mp3

# ボーカル音源から直接分離（vocal mode）
chorus-extract --mode vocal vocal.wav

# 複数ファイル・出力先指定
chorus-extract -o ./out -- song1.mp3 song2.flac
```

### 主要オプション

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--mode` | `full` | `full`=7ステム分離, `song`=ボーカル2段階, `vocal`=ボーカル1段階 |
| `--output-dir` / `-o` | `output` | 出力ディレクトリ |
| `--output-format` | `wav` | 出力形式 (`wav` / `mp3` / `flac` / `m4a` / `ogg`) |
| `--stage1-model` | モード依存 | Stage1 モデル名（`--list-models` で確認） |
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

```bash
chorus-extract --list-models
```

### Full mode（既定）

| キー | 用途 | 説明 |
|------|------|------|
| `htdemucs-6s` | Multi-stem（既定） | Demucs v4 — 6 ステム同時分離 |

### Song mode

| キー | 用途 | 説明 |
|------|------|------|
| `mel-roformer-vocals` | Stage1（既定） | MelBand Roformer by Kimberley Jensen（vocals SDR 12.6） |
| `bs-roformer-vocals` | Stage1 | BS-Roformer Viperx（vocals SDR 11.8） |

### Stage2（全モード共通）

| キー | 説明 |
|------|------|
| `mel-roformer-karaoke` | Mel-Band Roformer aufr33/viperx（既定） |
| `mel-roformer-karaoke-gabox` | MelBand Roformer by Gabox V2 |
| `mel-roformer-karaoke-becruily` | MelBand Roformer by becruily |
| `uvr-bve` | UVR BVE バッキングボーカル抽出 |
| `uvr-hp-karaoke` | UVR HP-Karaoke 6 |
| `uvr-mdx-karaoke` | UVR MDX-Net Karaoke 2 |

## 出力ファイル

### full mode（既定）

```
output/
├── {stem}_lead.wav       # リードボーカル
├── {stem}_chorus.wav     # ハモリ（バッキングボーカル）
├── {stem}_guitar.wav     # ギター
├── {stem}_bass.wav       # ベース
├── {stem}_drums.wav      # ドラム
├── {stem}_keyboard.wav   # キーボード / ピアノ
└── {stem}_other.wav      # その他
```

### song mode

```
output/
├── {stem}_lead.wav       # リードボーカル
└── {stem}_chorus.wav     # ハモリ（バッキングボーカル）

# --keep-intermediate 指定時
output/intermediate/
└── {stem}_(Vocals)_*.wav        # Stage1 中間ファイル
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
