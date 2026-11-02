"""Small desktop front end for the released same-work detector."""
from __future__ import annotations

import json
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
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoints" / "same-work-detector-v1.9"
DETECT_SCRIPT = ROOT / "scripts" / "detect_same_work.py"
DETECT_REQUIREMENTS = ROOT / "requirements-detector.txt"
REQUIRED_MODULES = ("numpy", "scipy", "sklearn", "mido", "joblib")


def missing_modules() -> list[str]:
    return [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]


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
    return min(file_count, cpu_limit, memory_limit, 12)


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
        self.status = tk.StringVar(value="Select a folder containing MIDI files.")
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.busy = False
        self.current_proc: subprocess.Popen | None = None
        self.cancel_requested = threading.Event()
        self.started_at: float | None = None

        frame = ttk.Frame(window, padding=28, style="App.TFrame")
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="MIDI Same-Work Detector", style="Title.TLabel").pack(anchor="w")
        ttk.Label(frame, text="Find MIDI files that may be versions of the same musical work.", style="Hint.TLabel").pack(anchor="w", pady=(5, 20))

        card = ttk.Frame(frame, padding=20, style="Card.TFrame")
        card.pack(fill="x")
        self._folder_row(card, "MIDI folder", self.input_dir, self.choose_input)
        self._folder_row(card, "Output folder", self.output_dir, self.choose_output)
        worker_row = ttk.Frame(card, style="Card.TFrame")
        worker_row.pack(fill="x", pady=7)
        ttk.Label(worker_row, text="CPU workers", width=20, style="Field.TLabel").pack(side="left")
        ttk.Combobox(worker_row, textvariable=self.worker_choice,
                     values=["Auto"] + [str(i) for i in range(1, available_cpus() + 1)],
                     state="readonly", width=12).pack(side="left", padx=7)
        ttk.Label(worker_row, text="Auto leaves memory headroom", style="Field.TLabel").pack(side="left")
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
        self.start([[sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
                    [sys.executable, "-m", "pip", "install", "-r", str(DETECT_REQUIREMENTS)]], "Installing dependencies")

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
        target = Path(target_text)
        if source.resolve() == target.resolve():
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
        workers = choose_workers(files) if self.worker_choice.get() == "Auto" else int(self.worker_choice.get())
        self.append(f"Found {files} MIDI files. Using {workers} worker(s) on {available_cpus()} available CPU(s); backend: exact.")
        self.start([[sys.executable, str(DETECT_SCRIPT), str(source), str(CHECKPOINT),
                    str(target), "--workers", str(workers), "--backend", "exact"]],
                   "Detecting", target)

    def open_output(self) -> None:
        path = Path(self.output_dir.get().strip())
        if not path.is_dir():
            messagebox.showinfo("No results yet", "The output folder has not been created yet.")
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
                        messagebox.showinfo("Detection complete", f"Results saved to:\n{result['output']}")
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
