"""Small desktop front end for the released same-work detector."""
from __future__ import annotations

import json
import csv
import base64
import hashlib
import importlib.util
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import tkinter as tk
import webbrowser
from collections import defaultdict
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoints" / "same-work-detector-v1.9"
DETECT_SCRIPT = ROOT / "scripts" / "detect_same_work.py"
DETECT_REQUIREMENTS = ROOT / "requirements-detector.txt"
REQUIRED_MODULES = ("numpy", "scipy", "sklearn", "mido", "joblib")


def missing_modules() -> list[str]:
    return [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]


def load_match_groups(folder: Path) -> tuple[dict, list[list[str]]]:
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    grouped: dict[str, list[str]] = defaultdict(list)
    with (folder / "components.csv").open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            grouped[row["component"]].append(row["file"])
    if sum(map(len, grouped.values())) != int(summary["files"]):
        raise ValueError("Result file count does not match the summary.")
    matches = [sorted(files, key=str.lower) for files in grouped.values() if len(files) > 1]
    matches.sort(key=lambda files: (-len(files), files[0].lower()))
    return summary, matches


def group_key(files: list[str]) -> str:
    return hashlib.sha256("\0".join(files).encode("utf-8")).hexdigest()[:16]


def load_review_marks(folder: Path) -> dict[str, str]:
    path = folder / "manual_review.csv"
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {row["group_id"]: row["review"] for row in csv.DictReader(handle)}


def save_review_marks(folder: Path, groups: list[list[str]], marks: dict[str, str]) -> None:
    path = folder / "manual_review.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("group_id", "review", "midi_files"))
        for files in groups:
            key = group_key(files)
            if key in marks:
                writer.writerow((key, marks[key], json.dumps(files, ensure_ascii=False)))


def launch_midi_file(path: Path) -> None:
    if sys.platform == "win32":
        os.startfile(path)  # type: ignore[attr-defined]
    elif os.environ.get("WSL_DISTRO_NAME"):
        windows_path = subprocess.check_output(["wslpath", "-w", str(path)], text=True).strip()
        escaped = windows_path.replace("'", "''")
        script = f"Start-Process -FilePath '{escaped}'"
        encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
        result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
        if result.returncode:
            raise OSError("Windows could not open the MIDI file. Check that a MIDI player is installed.")
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def count_midi(folder: Path) -> int:
    return sum(p.suffix.lower() in {".mid", ".midi"} for p in folder.rglob("*") if p.is_file())


def find_midi_folders() -> list[tuple[Path, int]]:
    """Scan common user locations without traversing an entire disk."""
    roots = [Path.home() / "datasets", ROOT / "data", Path.home() / "Music",
             Path.home() / "Desktop", Path.home() / "Downloads"]
    if sys.platform.startswith("linux") and Path("/mnt/c/Users").is_dir():
        roots.extend((Path("/mnt/c/Users/ASUS/Desktop"), Path("/mnt/c/Users/ASUS/Music")))
    totals: dict[Path, int] = {}
    for root in dict.fromkeys(path.resolve() for path in roots if path.is_dir()):
        scanned = 0
        for current, dirs, files in os.walk(root):
            scanned += 1
            current_path = Path(current)
            depth = len(current_path.relative_to(root).parts)
            if scanned > 12000 or depth >= 7:
                dirs[:] = []
            else:
                dirs[:] = [name for name in dirs if not name.startswith(".") and name not in
                           {"node_modules", "venv", ".venv", "__pycache__", "site-packages"}]
            direct = sum(Path(name).suffix.lower() in {".mid", ".midi"} for name in files)
            if not direct:
                continue
            # Count MIDI files in every ancestor so dataset-level folders appear.
            ancestor = current_path
            while True:
                totals[ancestor] = totals.get(ancestor, 0) + direct
                if ancestor == root:
                    break
                ancestor = ancestor.parent
    return sorted(((path, count) for path, count in totals.items() if count >= 2),
                  key=lambda item: (-item[1], str(item[0])))[:60]


def available_cpus() -> int:
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    if sys.platform.startswith("linux"):
        try:
            quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
            if quota != "max":
                cpus = min(cpus, max(1, int(quota) // int(period)))
        except (OSError, ValueError):
            pass
    return max(1, cpus)


def available_memory_gib() -> float | None:
    if sys.platform.startswith("linux"):
        available = None
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) / (1024 ** 2)
                    break
        except (OSError, ValueError):
            pass
        try:
            limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
            if limit != "max":
                current = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
                remaining = max(0, int(limit) - current) / (1024 ** 3)
                return min(available, remaining) if available is not None else remaining
        except (OSError, ValueError):
            pass
        return available
    elif sys.platform == "win32":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("memory_load", ctypes.c_ulong),
                            ("total_physical", ctypes.c_ulonglong), ("available_physical", ctypes.c_ulonglong),
                            ("total_page_file", ctypes.c_ulonglong), ("available_page_file", ctypes.c_ulonglong),
                            ("total_virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
                            ("available_extended_virtual", ctypes.c_ulonglong)]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.available_physical / (1024 ** 3)
        except (AttributeError, OSError):
            pass
    return None


def choose_workers(file_count: int) -> int:
    if file_count < 20:
        return 1
    cpu_limit = max(1, available_cpus() - 1)
    memory = available_memory_gib()
    memory_limit = max(1, int(memory - 2)) if memory is not None else 4
    return min(file_count, cpu_limit, memory_limit)


def parse_worker_choice(choice: str, file_count: int) -> int:
    choice = choice.strip()
    if choice.lower() == "auto":
        return choose_workers(file_count)
    if not choice.isdecimal() or int(choice) < 1:
        raise ValueError("Enter Auto or a positive whole number of CPU workers.")
    workers = int(choice)
    if workers > available_cpus():
        raise ValueError(f"This environment has {available_cpus()} available CPUs. Enter a number no greater than that.")
    return workers


def stop_process_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


class DetectorApp:
    def __init__(self, window: tk.Tk) -> None:
        self.window = window
        self.window.title("MIDI Same-Work Detector")
        self.window.geometry("820x610")
        self.window.minsize(690, 530)
        self.window.configure(bg="#f4f7fb")
        style = ttk.Style(window)
        style.theme_use("clam")
        style.configure("App.TFrame", background="#f4f7fb")
        style.configure("Card.TFrame", background="#ffffff")
        style.configure("Title.TLabel", background="#f4f7fb", foreground="#142438", font=("Arial", 21, "bold"))
        style.configure("Hint.TLabel", background="#f4f7fb", foreground="#607286", font=("Arial", 10))
        style.configure("Field.TLabel", background="#ffffff", foreground="#23364a", font=("Arial", 10, "bold"))
        style.configure("Status.TLabel", background="#f4f7fb", foreground="#244f67", font=("Arial", 10))
        style.configure("Accent.TButton", background="#167c90", foreground="white", font=("Arial", 10, "bold"), padding=(16, 9))
        style.map("Accent.TButton", background=[("active", "#12677a"), ("disabled", "#a9c8cd")])
        style.configure("Soft.TButton", background="#e9f0f5", foreground="#24455a", font=("Arial", 10), padding=(12, 9))
        style.map("Soft.TButton", background=[("active", "#dbeaf1")])
        style.configure("Folder.TEntry", padding=8, font=("Arial", 10))
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.worker_choice = tk.StringVar(value="Auto")
        self.mode_choice = tk.StringVar(value="Exact (v1.9)")
        self.status = tk.StringVar(value="Select a folder containing MIDI files.")
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.busy = False
        self.current_proc: subprocess.Popen | None = None
        self.cancel_requested = threading.Event()
        self.started_at: float | None = None
        self.latest_output: Path | None = None

        frame = ttk.Frame(window, padding=28, style="App.TFrame")
        frame.pack(fill="both", expand=True)
        header = ttk.Frame(frame, style="App.TFrame")
        header.pack(fill="x")
        ttk.Label(header, text="MIDI Same-Work Detector", style="Title.TLabel").pack(side="left")
        ttk.Button(header, text="Exit", command=self.close, style="Soft.TButton").pack(side="right")
        ttk.Button(header, text="View matches", command=self.show_matches, style="Soft.TButton").pack(side="right", padx=8)
        ttk.Label(frame, text="Find MIDI files that may be versions of the same musical work.", style="Hint.TLabel").pack(anchor="w", pady=(5, 20))

        card = ttk.Frame(frame, padding=20, style="Card.TFrame")
        card.pack(fill="x")
        self._folder_row(card, "MIDI folder", self.input_dir, self.choose_input)
        self._folder_row(card, "Output folder", self.output_dir, self.choose_output)
        worker_row = ttk.Frame(card, style="Card.TFrame")
        worker_row.pack(fill="x", pady=7)
        ttk.Label(worker_row, text="CPU workers", width=20, style="Field.TLabel").pack(side="left")
        ttk.Entry(worker_row, textvariable=self.worker_choice, style="Folder.TEntry", width=12).pack(side="left", padx=7)
        ttk.Label(worker_row, text="Auto or a number (available: " + str(available_cpus()) + ")", style="Field.TLabel").pack(side="left")
        mode_row = ttk.Frame(card, style="Card.TFrame")
        mode_row.pack(fill="x", pady=7)
        ttk.Label(mode_row, text="Detection mode", width=20, style="Field.TLabel").pack(side="left")
        ttk.Combobox(mode_row, textvariable=self.mode_choice,
                     values=("Exact (v1.9)", "Fast (v1.9 + FAISS)"),
                     state="readonly", width=27).pack(side="left", padx=7)
        ttk.Label(frame, text="Includes .mid and .midi files in subfolders. Install dependencies before the first run.", style="Hint.TLabel").pack(anchor="w", pady=(13, 14))

        buttons = ttk.Frame(frame, style="App.TFrame")
        buttons.pack(fill="x")
        self.install_button = ttk.Button(buttons, text="Install dependencies", command=self.install, style="Soft.TButton")
        self.install_button.pack(side="left")
        self.run_button = ttk.Button(buttons, text="Start detection", command=self.run, style="Accent.TButton")
        self.run_button.pack(side="left", padx=10)
        self.cancel_button = ttk.Button(buttons, text="Cancel detection", command=self.cancel, style="Soft.TButton", state="disabled")
        self.cancel_button.pack(side="left")
        ttk.Button(buttons, text="Open results", command=self.open_output, style="Soft.TButton").pack(side="left", padx=10)
        ttk.Button(buttons, text="Find MIDI folders", command=self.search_folders, style="Soft.TButton").pack(side="left")
        ttk.Label(frame, textvariable=self.status, style="Status.TLabel", wraplength=730).pack(anchor="w", pady=(18, 8))

        log_frame = ttk.Frame(frame, style="App.TFrame")
        log_frame.pack(fill="both", expand=True)
        self.log = tk.Text(log_frame, height=12, wrap="word", state="disabled", bg="#15283a", fg="#d9e9ee", insertbackground="white", relief="flat", padx=13, pady=11, font=("Consolas", 10))
        self.log.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(log_frame, command=self.log.yview)
        scrollbar.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=scrollbar.set)
        window.protocol("WM_DELETE_WINDOW", self.close)
        window.after(100, self.poll)

    def _folder_row(self, parent: ttk.Frame, label: str, variable: tk.StringVar, action) -> None:
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", pady=7)
        ttk.Label(row, text=label, width=20, style="Field.TLabel").pack(side="left")
        ttk.Entry(row, textvariable=variable, style="Folder.TEntry").pack(side="left", fill="x", expand=True, padx=7)
        ttk.Button(row, text="Browse...", command=action, style="Soft.TButton").pack(side="right")

    def choose_input(self) -> None:
        selected = filedialog.askdirectory(title="Select MIDI folder")
        if selected:
            self.input_dir.set(selected)
            if not self.output_dir.get():
                self.output_dir.set(str(Path(selected).parent / (Path(selected).name + "_detector_results")))

    def choose_output(self) -> None:
        selected = filedialog.askdirectory(title="Select output folder")
        if selected:
            self.output_dir.set(selected)

    def search_folders(self) -> None:
        self.status.set("Searching common folders for MIDI files...")
        threading.Thread(target=lambda: self.events.put(("folders", find_midi_folders())), daemon=True).start()

    def show_folders(self, folders: list[tuple[Path, int]]) -> None:
        if not folders:
            self.status.set("No MIDI folders found in common locations. Use Browse to choose another folder.")
            return
        self.status.set(f"Found {len(folders)} MIDI folders. Select one from the list.")
        dialog = tk.Toplevel(self.window)
        dialog.title("Found MIDI folders")
        dialog.geometry("780x430")
        dialog.configure(bg="#f4f7fb")
        dialog.transient(self.window)
        frame = ttk.Frame(dialog, padding=18, style="App.TFrame")
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Choose a MIDI folder", style="Title.TLabel").pack(anchor="w", pady=(0, 12))
        listbox = tk.Listbox(frame, font=("Arial", 10), bg="white", fg="#23364a",
                             selectbackground="#167c90", selectforeground="white")
        listbox.pack(fill="both", expand=True)
        for path, count in folders:
            listbox.insert("end", f"{count:>5} MIDI   {path}")

        def select(_event=None) -> None:
            selection = listbox.curselection()
            if not selection:
                return
            path = folders[selection[0]][0]
            self.input_dir.set(str(path))
            self.output_dir.set(str(path.parent / (path.name + "_detector_results")))
            dialog.destroy()

        ttk.Button(frame, text="Use selected folder", command=select, style="Accent.TButton").pack(anchor="e", pady=(12, 0))
        listbox.bind("<Double-Button-1>", select)

    def append(self, line: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def start(self, commands: list[list[str]], title: str, output: Path | None = None) -> None:
        if self.busy:
            return
        self.busy = True
        self.started_at = time.monotonic()
        self.cancel_requested.clear()
        self.install_button.configure(state="disabled")
        self.run_button.configure(state="disabled")
        self.cancel_button.configure(state="normal" if output is not None else "disabled")
        self.status.set(title + " - please wait...")
        self.append(title)

        def worker() -> None:
            try:
                code = 0
                child_env = os.environ.copy()
                child_env["PYTHONPATH"] = str(ROOT) + os.pathsep + child_env.get("PYTHONPATH", "")
                for cmd in commands:
                    if self.cancel_requested.is_set():
                        code = 130
                        break
                    proc = subprocess.Popen(cmd, cwd=ROOT, env=child_env, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True,
                                            errors="replace", bufsize=1,
                                            start_new_session=(sys.platform != "win32"),
                                            creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0))
                    self.current_proc = proc
                    if self.cancel_requested.is_set():
                        stop_process_tree(proc)
                    assert proc.stdout is not None
                    for line in proc.stdout:
                        self.events.put(("log", line.rstrip()))
                    code = proc.wait()
                    self.current_proc = None
                    if code or self.cancel_requested.is_set():
                        break
                self.events.put(("done", json.dumps({"code": code, "title": title,
                                                       "output": str(output) if output else "",
                                                       "cancelled": self.cancel_requested.is_set()})))
            except Exception as exc:
                self.current_proc = None
                self.events.put(("done", json.dumps({"code": -1, "title": title,
                                                       "error": str(exc)})))

        threading.Thread(target=worker, daemon=True).start()

    def cancel(self) -> None:
        if not self.busy or self.cancel_requested.is_set():
            return
        self.cancel_requested.set()
        self.cancel_button.configure(state="disabled")
        self.status.set("Stopping detection...")
        self.append("Cancellation requested. Waiting for detector processes to stop...")
        if self.current_proc is not None:
            threading.Thread(target=stop_process_tree, args=(self.current_proc,), daemon=True).start()

    def close(self) -> None:
        if self.busy and self.current_proc is not None:
            self.cancel_requested.set()
            stop_process_tree(self.current_proc)
        self.window.destroy()

    def install(self) -> None:
        if not self.supported_python():
            return
        commands = [[sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
                    [sys.executable, "-m", "pip", "install", "-r", str(DETECT_REQUIREMENTS)]]
        if self.mode_choice.get() == "Fast (v1.9 + FAISS)":
            commands.append([sys.executable, "-m", "pip", "install", "faiss-cpu>=1.12,<2"])
        self.start(commands, "Installing dependencies")

    def supported_python(self) -> bool:
        if sys.version_info >= (3, 10):
            return True
        if messagebox.askyesno("Python version too old", "Python 3.10 or newer is required.\nCurrent version: " + sys.version.split()[0] + "\n\nOpen the Python download page?"):
            webbrowser.open("https://www.python.org/downloads/windows/")
        return False

    def run(self) -> None:
        if not self.supported_python():
            return
        source = Path(self.input_dir.get().strip())
        target_text = self.output_dir.get().strip()
        if not source.is_dir() or not target_text:
            messagebox.showerror("Choose folders", "Select a MIDI folder and an output folder.")
            return
        output_root = Path(target_text)
        if source.resolve() == output_root.resolve():
            messagebox.showerror("Invalid output folder", "The output folder must differ from the MIDI folder.")
            return
        files = count_midi(source)
        if files < 2:
            messagebox.showerror("Not enough files", "At least two .mid or .midi files are required.")
            return
        if not CHECKPOINT.is_dir():
            messagebox.showerror("Checkpoint missing", f"Detector checkpoint not found: {CHECKPOINT}")
            return
        missing = missing_modules()
        if missing:
            messagebox.showerror("Dependencies missing", "Missing: " + ", ".join(missing) + "\nClick Install dependencies first.")
            return
        backend = "faiss" if self.mode_choice.get() == "Fast (v1.9 + FAISS)" else "exact"
        target = output_root / ("fast_v1.9_faiss" if backend == "faiss" else "exact_v1.9")
        if backend == "faiss" and importlib.util.find_spec("faiss") is None:
            messagebox.showerror("FAISS missing", "Fast mode needs faiss-cpu. Click Install dependencies first, or select Exact mode.")
            return
        try:
            workers = parse_worker_choice(self.worker_choice.get(), files)
        except ValueError as exc:
            messagebox.showerror("Invalid worker count", str(exc))
            return
        self.append(f"Found {files} MIDI files. Using {workers} worker(s) on {available_cpus()} available CPU(s); backend: {backend}.")
        self.latest_output = target
        self.start([[sys.executable, str(DETECT_SCRIPT), str(source), str(CHECKPOINT),
                    str(target), "--workers", str(workers), "--backend", backend]],
                   "Detecting", target)

    def open_output(self) -> None:
        path = self.latest_output or Path(self.output_dir.get().strip())
        if not path.is_dir() or not (path / "summary.json").is_file():
            messagebox.showinfo("No completed results", "No completed result files were found. Check the status and log in this window; a cancelled or failed run does not produce a complete result.")
            return
        try:
            if sys.platform == "win32":
                os.startfile(path)  # type: ignore[attr-defined]
            elif os.environ.get("WSL_DISTRO_NAME"):
                windows_path = subprocess.check_output(["wslpath", "-w", str(path)], text=True).strip()
                subprocess.Popen(["explorer.exe", windows_path], stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except (OSError, subprocess.SubprocessError) as exc:
            messagebox.showerror("Cannot open folder", f"Open this path manually:\n{path}\n\n{exc}")

    def show_matches(self) -> None:
        folder = self.latest_output
        if folder is None:
            root = Path(self.output_dir.get().strip()) if self.output_dir.get().strip() else None
            if root is not None:
                mode = "fast_v1.9_faiss" if self.mode_choice.get() == "Fast (v1.9 + FAISS)" else "exact_v1.9"
                folder = root / mode
        if folder is None or not (folder / "summary.json").is_file():
            messagebox.showinfo("No completed results", "Run a detection first, or select an output folder containing completed results.")
            return
        try:
            summary, groups = load_match_groups(folder)
            marks = load_review_marks(folder)
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            messagebox.showerror("Cannot read results", str(exc))
            return

        dialog = tk.Toplevel(self.window)
        dialog.title("Matched works")
        dialog.geometry("960x650")
        dialog.minsize(650, 420)
        dialog.configure(bg="#f4f7fb")
        frame = ttk.Frame(dialog, padding=18, style="App.TFrame")
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Matched works", style="Title.TLabel").pack(anchor="w")
        matched_files = sum(map(len, groups))
        total = int(summary["files"])
        detail = (f"{len(groups)} groups  |  {matched_files} files in groups  |  "
                  f"{total - matched_files} files without a match  |  {total} total")
        ttk.Label(frame, text=detail, style="Hint.TLabel").pack(anchor="w", pady=(5, 5))
        ttk.Label(frame, text="Double-click a MIDI file to open it. Review marks are saved separately from detector results.",
                  style="Hint.TLabel").pack(anchor="w", pady=(0, 12))
        search = tk.StringVar()
        search_row = ttk.Frame(frame, style="App.TFrame")
        search_row.pack(fill="x", pady=(0, 9))
        ttk.Label(search_row, text="Find file:", style="Status.TLabel").pack(side="left")
        ttk.Entry(search_row, textvariable=search, style="Folder.TEntry").pack(side="left", fill="x", expand=True, padx=8)
        tree_frame = ttk.Frame(frame, style="App.TFrame")
        tree_frame.pack(fill="both", expand=True)
        tree = ttk.Treeview(tree_frame, columns=("count", "review", "file"), show="tree headings")
        tree.heading("#0", text="Matched group")
        tree.heading("count", text="Files")
        tree.heading("review", text="Manual review")
        tree.heading("file", text="MIDI file (relative path)")
        tree.column("#0", width=130, stretch=False)
        tree.column("count", width=70, anchor="center", stretch=False)
        tree.column("review", width=140, stretch=False)
        tree.column("file", width=570)
        tree.tag_configure("group", background="#dceef1", foreground="#173a48")
        tree.tag_configure("confirmed", background="#dbf0df", foreground="#175b32")
        tree.tag_configure("flagged", background="#fff0d3", foreground="#7b4d10")
        tree.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        scrollbar.pack(side="right", fill="y")
        tree.configure(yscrollcommand=scrollbar.set)

        item_files: dict[str, str] = {}
        item_groups: dict[str, int] = {}

        def populate(*_args) -> None:
            tree.delete(*tree.get_children())
            item_files.clear()
            item_groups.clear()
            query = search.get().strip().casefold()
            for number, files in enumerate(groups, 1):
                if query and not any(query in name.casefold() for name in files):
                    continue
                mark = marks.get(group_key(files), "Not reviewed")
                tag = "confirmed" if mark == "Confirmed" else "flagged" if mark == "Flagged" else "group"
                parent = tree.insert("", "end", text=f"Group {number}", values=(len(files), mark, ""),
                                     tags=(tag,), open=bool(query) or number <= 8)
                item_groups[parent] = number - 1
                for name in files:
                    child = tree.insert(parent, "end", text="", values=("", "", name))
                    item_files[child] = name
                    item_groups[child] = number - 1

        def open_selected(_event=None) -> None:
            selection = tree.selection()
            if not selection or selection[0] not in item_files:
                messagebox.showinfo("Select a MIDI", "Select a file inside a group, then click Open MIDI.", parent=dialog)
                return
            relative = Path(item_files[selection[0]])
            source = Path(self.input_dir.get().strip())
            candidate = (source / relative).resolve()
            if not source.is_dir() or not candidate.is_file():
                selected = filedialog.askdirectory(title="Find the original MIDI folder", parent=dialog)
                if not selected:
                    return
                source = Path(selected)
                self.input_dir.set(str(source))
                candidate = (source / relative).resolve()
            try:
                candidate.relative_to(source.resolve())
                if not candidate.is_file():
                    raise FileNotFoundError(candidate)
                launch_midi_file(candidate)
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                messagebox.showerror("Cannot open MIDI", f"File: {candidate}\n\n{exc}", parent=dialog)

        def mark_selected(mark: str) -> None:
            selection = tree.selection()
            if not selection or selection[0] not in item_groups:
                messagebox.showinfo("Select a group", "Select a group or one of its MIDI files first.", parent=dialog)
                return
            key = group_key(groups[item_groups[selection[0]]])
            if mark == "Not reviewed":
                marks.pop(key, None)
            else:
                marks[key] = mark
            try:
                save_review_marks(folder, groups, marks)
            except OSError as exc:
                messagebox.showerror("Cannot save review", str(exc), parent=dialog)
                return
            populate()

        actions = ttk.Frame(frame, style="App.TFrame")
        actions.pack(fill="x", pady=(12, 0))
        ttk.Button(actions, text="Open MIDI", command=open_selected, style="Accent.TButton").pack(side="left")
        ttk.Button(actions, text="Confirm group", command=lambda: mark_selected("Confirmed"), style="Soft.TButton").pack(side="left", padx=8)
        ttk.Button(actions, text="Flag group", command=lambda: mark_selected("Flagged"), style="Soft.TButton").pack(side="left")
        ttk.Button(actions, text="Clear mark", command=lambda: mark_selected("Not reviewed"), style="Soft.TButton").pack(side="left", padx=8)
        ttk.Label(actions, text="Saved to manual_review.csv", style="Hint.TLabel").pack(side="right")

        search.trace_add("write", populate)
        tree.bind("<Double-Button-1>", open_selected)
        populate()

    def poll(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "folders":
                    self.show_folders(payload)
                    continue
                if kind == "log":
                    self.append(str(payload))
                    continue
                result = json.loads(str(payload))
                self.busy = False
                self.started_at = None
                self.install_button.configure(state="normal")
                self.run_button.configure(state="normal")
                self.cancel_button.configure(state="disabled")
                if result.get("cancelled"):
                    self.status.set("Detection cancelled.")
                    self.append("Detection cancelled. Any files already in the output folder may be incomplete.")
                elif result["code"]:
                    self.status.set("Failed. Check the log below.")
                    if result.get("error"):
                        self.append(result["error"])
                    messagebox.showerror("Failed", "Check the log below. If a module is missing, click Install dependencies first.")
                else:
                    self.status.set("Done. " + ("Results saved." if result.get("output") else "Dependencies installed."))
                    if result.get("output"):
                        self.append("Results: components.csv (groups), relation_edges.csv (pair evidence), summary.json (counts).")
                        self.show_matches()
        except queue.Empty:
            pass
        if self.busy and self.started_at is not None and not self.cancel_requested.is_set():
            elapsed = int(time.monotonic() - self.started_at)
            self.status.set(f"Running... {elapsed // 60}m {elapsed % 60:02d}s elapsed. Large datasets can take a while.")
        self.window.after(100, self.poll)


def main() -> None:
    window = tk.Tk()
    DetectorApp(window)
    window.mainloop()


if __name__ == "__main__":
    main()
