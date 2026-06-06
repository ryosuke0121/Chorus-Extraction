"""ドメイン例外定義。すべて ChorusExtractionError を基底とし、意味ある診断メッセージを持つ。"""

from __future__ import annotations


class ChorusExtractionError(Exception):
    """コーラス抽出ツール共通基底例外。"""


class InvalidInputError(ChorusExtractionError):
    """入力ファイルが無効（存在しない・対応フォーマット外・空ファイル等）。"""


class ModelDownloadError(ChorusExtractionError):
    """モデル重みのダウンロードまたはロードに失敗した。"""


class SeparationError(ChorusExtractionError):
    """音源分離処理中にエラーが発生した。"""


class DeviceUnavailableError(ChorusExtractionError):
    """指定されたデバイス（CUDA 等）が利用できない。"""
