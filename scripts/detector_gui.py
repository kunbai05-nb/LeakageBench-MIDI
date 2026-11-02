"""Small desktop front end for the released same-work detector."""
from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoints" / "same-work-detector-v1.9"
DETECT_SCRIPT = ROOT / "scripts" / "detect_same_work.py"


def count_midi(folder: Path) -> int:
    return sum(p.suffix.lower() in {".mid", ".midi"} for p in folder.rglob("*") if p.is_file())


class DetectorApp:
    def __init__(self, window: tk.Tk) -> None:
        self.window = window
        self.window.title("MIDI 同作品检测器")
        self.window.geometry("820x610")
        self.window.minsize(690, 530)
        self.window.configure(bg="#f4f7fb")
        style = ttk.Style(window)
        style.theme_use("clam")
        style.configure("App.TFrame", background="#f4f7fb")
        style.configure("Card.TFrame", background="#ffffff")
        style.configure("Title.TLabel", background="#f4f7fb", foreground="#142438", font=("Microsoft YaHei UI", 21, "bold"))
        style.configure("Hint.TLabel", background="#f4f7fb", foreground="#607286", font=("Microsoft YaHei UI", 10))
        style.configure("Field.TLabel", background="#ffffff", foreground="#23364a", font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("Status.TLabel", background="#f4f7fb", foreground="#244f67", font=("Microsoft YaHei UI", 10))
        style.configure("Accent.TButton", background="#167c90", foreground="white", font=("Microsoft YaHei UI", 10, "bold"), padding=(16, 9))
        style.map("Accent.TButton", background=[("active", "#12677a"), ("disabled", "#a9c8cd")])
        style.configure("Soft.TButton", background="#e9f0f5", foreground="#24455a", font=("Microsoft YaHei UI", 10), padding=(12, 9))
        style.map("Soft.TButton", background=[("active", "#dbeaf1")])
        style.configure("Folder.TEntry", padding=8, font=("Microsoft YaHei UI", 10))
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.status = tk.StringVar(value="选择包含 MIDI 文件的文件夹。")
        self.events: queue.Queue[tuple[str, str]] = queue.Queue()
        self.busy = False

        frame = ttk.Frame(window, padding=28, style="App.TFrame")
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="MIDI 同作品检测", style="Title.TLabel").pack(anchor="w")
        ttk.Label(frame, text="选一个文件夹，自动找出可能属于同一作品的不同 MIDI 版本。", style="Hint.TLabel").pack(anchor="w", pady=(5, 20))

        card = ttk.Frame(frame, padding=20, style="Card.TFrame")
        card.pack(fill="x")
        self._folder_row(card, "① 选择 MIDI 文件夹", self.input_dir, self.choose_input)
        self._folder_row(card, "② 选择结果位置", self.output_dir, self.choose_output)
        ttk.Label(frame, text="支持子文件夹中的 .mid / .midi 文件。首次使用请先安装运行环境。", style="Hint.TLabel").pack(anchor="w", pady=(13, 14))

        buttons = ttk.Frame(frame, style="App.TFrame")
        buttons.pack(fill="x")
        self.install_button = ttk.Button(buttons, text="安装运行环境（首次）", command=self.install, style="Soft.TButton")
        self.install_button.pack(side="left")
        self.run_button = ttk.Button(buttons, text="开始检测", command=self.run, style="Accent.TButton")
        self.run_button.pack(side="left", padx=10)
        ttk.Button(buttons, text="查看结果", command=self.open_output, style="Soft.TButton").pack(side="left")
        ttk.Label(frame, textvariable=self.status, style="Status.TLabel", wraplength=730).pack(anchor="w", pady=(18, 8))

        log_frame = ttk.Frame(frame, style="App.TFrame")
        log_frame.pack(fill="both", expand=True)
        self.log = tk.Text(log_frame, height=12, wrap="word", state="disabled", bg="#15283a", fg="#d9e9ee", insertbackground="white", relief="flat", padx=13, pady=11, font=("Consolas", 10))
        self.log.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(log_frame, command=self.log.yview)
        scrollbar.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=scrollbar.set)
        window.after(100, self.poll)

    def _folder_row(self, parent: ttk.Frame, label: str, variable: tk.StringVar, action) -> None:
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", pady=7)
        ttk.Label(row, text=label, width=20, style="Field.TLabel").pack(side="left")
        ttk.Entry(row, textvariable=variable, style="Folder.TEntry").pack(side="left", fill="x", expand=True, padx=7)
        ttk.Button(row, text="浏览…", command=action, style="Soft.TButton").pack(side="right")

    def choose_input(self) -> None:
        selected = filedialog.askdirectory(title="选择 MIDI 文件夹")
        if selected:
            self.input_dir.set(selected)
            if not self.output_dir.get():
                self.output_dir.set(str(Path(selected).parent / (Path(selected).name + "_检测结果")))

    def choose_output(self) -> None:
        selected = filedialog.askdirectory(title="选择结果文件夹")
        if selected:
            self.output_dir.set(selected)

    def append(self, line: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def start(self, commands: list[list[str]], title: str, output: Path | None = None) -> None:
        if self.busy:
            return
        self.busy = True
        self.install_button.configure(state="disabled")
        self.run_button.configure(state="disabled")
        self.status.set(title + "，请等待…")
        self.append(title)

        def worker() -> None:
            try:
                code = 0
                for cmd in commands:
                    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True,
                                            errors="replace", bufsize=1)
                    assert proc.stdout is not None
                    for line in proc.stdout:
                        self.events.put(("log", line.rstrip()))
                    code = proc.wait()
                    if code:
                        break
                self.events.put(("done", json.dumps({"code": code, "title": title,
                                                       "output": str(output) if output else ""})))
            except Exception as exc:
                self.events.put(("done", json.dumps({"code": -1, "title": title,
                                                       "error": str(exc)})))

        threading.Thread(target=worker, daemon=True).start()

    def install(self) -> None:
        if not self.supported_python():
            return
        self.start([[sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
                    [sys.executable, "-m", "pip", "install", str(ROOT)]], "安装运行环境")

    def supported_python(self) -> bool:
        if sys.version_info >= (3, 10):
            return True
        if messagebox.askyesno("Python 版本过旧", "检测器需要 Python 3.10 或更新版本。\n当前版本：" + sys.version.split()[0] + "\n\n打开 Python 下载页面？"):
            webbrowser.open("https://www.python.org/downloads/windows/")
        return False

    def run(self) -> None:
        if not self.supported_python():
            return
        source = Path(self.input_dir.get().strip())
        target_text = self.output_dir.get().strip()
        if not source.is_dir() or not target_text:
            messagebox.showerror("请选择文件夹", "请选择 MIDI 文件夹和结果位置。")
            return
        target = Path(target_text)
        if source.resolve() == target.resolve():
            messagebox.showerror("结果位置错误", "结果文件夹不能与 MIDI 文件夹相同。")
            return
        files = count_midi(source)
        if files < 2:
            messagebox.showerror("文件不足", "至少需要两个 .mid 或 .midi 文件。")
            return
        if not CHECKPOINT.is_dir():
            messagebox.showerror("模型缺失", f"找不到模型：{CHECKPOINT}")
            return
        self.append(f"找到 {files} 个 MIDI 文件。规模越大，检测时间越长。")
        self.start([[sys.executable, str(DETECT_SCRIPT), str(source), str(CHECKPOINT),
                    str(target), "--workers", "1", "--backend", "exact"]],
                   "检测中", target)

    def open_output(self) -> None:
        path = Path(self.output_dir.get().strip())
        if not path.is_dir():
            messagebox.showinfo("暂无结果", "结果文件夹尚未生成。")
            return
        import os
        if sys.platform == "win32":
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])

    def poll(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    self.append(payload)
                    continue
                result = json.loads(payload)
                self.busy = False
                self.install_button.configure(state="normal")
                self.run_button.configure(state="normal")
                if result["code"]:
                    self.status.set("执行失败。请查看下方信息。")
                    if result.get("error"):
                        self.append(result["error"])
                    messagebox.showerror("执行失败", "请查看窗口下方的信息；若提示缺少模块，先点击安装依赖。")
                else:
                    self.status.set("完成。" + ("结果已保存。" if result.get("output") else "依赖已安装。"))
                    if result.get("output"):
                        self.append("结果：components.csv（分组）、relation_edges.csv（配对证据）、summary.json（统计）。")
                        messagebox.showinfo("检测完成", f"结果已保存到：\n{result['output']}")
        except queue.Empty:
            pass
        self.window.after(100, self.poll)


def main() -> None:
    window = tk.Tk()
    DetectorApp(window)
    window.mainloop()


if __name__ == "__main__":
    main()
