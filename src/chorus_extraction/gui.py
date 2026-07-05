"""tkinter ベースのデスクトップ GUI。CLI と同じ RunConfig / extract を呼ぶ薄いラッパー。"""

from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from chorus_extraction.config import RunConfig, build_run_config
from chorus_extraction.errors import ChorusExtractionError
from chorus_extraction.pipeline import ExtractionResult, extract
from chorus_extraction.separator_runner import make_separator, run_separation

_MODES = ("full", "song", "vocal")
_FORMATS = ("wav", "mp3", "flac", "m4a", "ogg")
_DEVICES = ("auto", "cuda", "cpu")

_APP_DIR = Path.home() / ".chorus-extraction"


class _QueueHandler(logging.Handler):
    """ログレコードをキューに積むだけのハンドラ（GUI スレッドがポーリングして表示）。"""

    def __init__(self, log_queue: queue.Queue[str]) -> None:
        super().__init__()
        self._queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        self._queue.put(self.format(record))


class App:
    """メインウィンドウ。"""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Chorus Extraction")
        root.geometry("640x520")
        root.minsize(520, 420)

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.worker: threading.Thread | None = None

        frame = ttk.Frame(root, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        # --- 入力ファイル ---
        ttk.Label(frame, text="入力ファイル").pack(anchor=tk.W)
        list_row = ttk.Frame(frame)
        list_row.pack(fill=tk.X)
        self.file_list = tk.Listbox(list_row, height=4)
        self.file_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        btns = ttk.Frame(list_row)
        btns.pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(btns, text="追加...", command=self._add_files).pack(fill=tk.X)
        ttk.Button(btns, text="クリア", command=lambda: self.file_list.delete(0, tk.END)).pack(
            fill=tk.X, pady=(4, 0)
        )

        # --- オプション ---
        opts = ttk.Frame(frame)
        opts.pack(fill=tk.X, pady=8)
        self.mode_var = tk.StringVar(value="full")
        self.format_var = tk.StringVar(value="wav")
        self.device_var = tk.StringVar(value="auto")
        options: tuple[tuple[str, tk.StringVar, tuple[str, ...]], ...] = (
            ("モード", self.mode_var, _MODES),
            ("出力形式", self.format_var, _FORMATS),
            ("デバイス", self.device_var, _DEVICES),
        )
        for col, (label, var, values) in enumerate(options):
            ttk.Label(opts, text=label).grid(row=0, column=col, sticky=tk.W, padx=(0, 12))
            ttk.Combobox(opts, textvariable=var, values=values, state="readonly", width=8).grid(
                row=1, column=col, sticky=tk.W, padx=(0, 12)
            )

        # --- 出力先 ---
        ttk.Label(frame, text="出力フォルダ").pack(anchor=tk.W)
        out_row = ttk.Frame(frame)
        out_row.pack(fill=tk.X)
        self.output_var = tk.StringVar(value=str(Path.home() / "Music" / "chorus-extract"))
        ttk.Entry(out_row, textvariable=self.output_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(out_row, text="参照...", command=self._choose_output).pack(side=tk.LEFT, padx=(6, 0))

        # --- 実行 ---
        self.run_button = ttk.Button(frame, text="抽出開始", command=self._start)
        self.run_button.pack(pady=8)

        # --- ログ ---
        self.log_text = scrolledtext.ScrolledText(frame, height=12, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self._setup_logging()
        root.after(100, self._poll_log)

    # ------------------------------------------------------------------
    def _setup_logging(self) -> None:
        handler = _QueueHandler(self.log_queue)
        handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(handler)

    def _poll_log(self) -> None:
        try:
            while True:
                line = self.log_queue.get_nowait()
                self.log_text.configure(state=tk.NORMAL)
                self.log_text.insert(tk.END, line + "\n")
                self.log_text.see(tk.END)
                self.log_text.configure(state=tk.DISABLED)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log)

    def _add_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="音声ファイルを選択",
            filetypes=[
                ("音声ファイル", "*.wav *.mp3 *.flac *.m4a *.ogg *.aiff *.aif *.wma"),
                ("すべて", "*.*"),
            ],
        )
        for p in paths:
            self.file_list.insert(tk.END, p)

    def _choose_output(self) -> None:
        chosen = filedialog.askdirectory(title="出力フォルダを選択")
        if chosen:
            self.output_var.set(chosen)

    # ------------------------------------------------------------------
    def _start(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        inputs = [Path(p) for p in self.file_list.get(0, tk.END)]
        if not inputs:
            messagebox.showwarning("入力なし", "入力ファイルを追加してください。")
            return

        try:
            cfg = build_run_config(
                inputs=inputs,
                mode=self.mode_var.get(),
                output_dir=Path(self.output_var.get()),
                output_format=self.format_var.get(),
                stage1_model_name=None,
                stage2_model_name=None,
                model_dir=_APP_DIR / "models",
                keep_intermediate=False,
                device=self.device_var.get(),
                lead_name_template="{stem}_lead",
                chorus_name_template="{stem}_chorus",
                verbosity=1,
            )
        except ChorusExtractionError as exc:
            messagebox.showerror("設定エラー", str(exc))
            return

        self.run_button.configure(state=tk.DISABLED, text="処理中...")
        self.worker = threading.Thread(target=self._run, args=(cfg,), daemon=True)
        self.worker.start()

    def _run(self, cfg: RunConfig) -> None:
        logger = logging.getLogger(__name__)
        try:
            separator = make_separator(
                output_dir=cfg.output_dir,
                output_format=cfg.output_format,
                model_dir=cfg.model_dir,
                device=cfg.device,
            )
            logger.info("処理を開始します（初回はモデルのダウンロードに時間がかかります）")
            results = extract(cfg, separator=separator, separate_fn=run_separation)
            for r in results:
                self._log_result(logger, r)
            logger.info("完了しました。出力先: %s", cfg.output_dir)
        except ChorusExtractionError as exc:
            logger.error("処理エラー: %s", exc)
        except Exception:
            logger.exception("予期しないエラーが発生しました")
        finally:
            self.root.after(0, lambda: self.run_button.configure(state=tk.NORMAL, text="抽出開始"))

    @staticmethod
    def _log_result(logger: logging.Logger, result: ExtractionResult) -> None:
        logger.info("[OK] %s", result.input_path.name)
        if result.lead_path:
            logger.info("  リード: %s", result.lead_path.name)
        if result.chorus_path:
            logger.info("  ハモリ: %s", result.chorus_path.name)
        for label, path in result.stems.items():
            logger.info("  %s: %s", label, path.name)


def main() -> None:
    """GUI エントリポイント。"""
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
