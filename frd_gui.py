# Copyright (c) 2026 Alessio Turchi (INAF - Osservatorio Astrofisico di Arcetri)
# Based on the FRD IDL package and its Python port (frd_toolkit.py),
# Copyright (c) Andrea Tozzi (INAF - Osservatorio Astrofisico di Arcetri)
# SPDX-License-Identifier: GPL-3.0-or-later (see LICENSE)

"""Minimal Tkinter GUI for the FRD pipeline: edit parameters/paths, load/save YAML, run."""

import logging
import logging.handlers
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import yaml

from lib import frd_pipeline

DEFAULT_CFG = Path(__file__).with_name("frd_config.yaml")

# (section, key, label, kind); kind: dir, file, str, float, optfloat, int, bool,
# floatlist, choice:a,b
FIELDS = [
    ("paths", "input_dir", "Input directory", "dir"),
    ("paths", "pattern", "File pattern", "str"),
    ("paths", "output_dir", "Output directory", "dir"),
    ("paths", "dark_file", "Dark FITS (optional)", "file"),
    ("distances", "use_header", "Use header distance", "bool"),
    ("distances", "header_key", "Header keyword", "str"),
    ("distances", "step_mm", "Step [mm] (no header)", "float"),
    ("distances", "offset_mm", "Offset added to distances [mm]", "float"),
    ("detector", "pix_size_mm", "Pixel size [mm]", "float"),
    ("pupil", "sogl", "1st threshold (fraction P-V)", "float"),
    ("pupil", "s2_p2v", "2nd threshold (empty = mean)", "optfloat"),
    ("frd_curve", "mode", "Integration centre", "choice:pupil,ccd"),
    ("frd_curve", "fn_max", "Max output F/#", "float"),
    ("frd_curve", "n_points", "Points per curve", "int"),
    ("frd_curve", "border_pix", "CCD border [pix]", "float"),
    ("frd_curve", "ee_levels", "EE levels (comma separated)", "floatlist"),
    ("plots", "format", "Plot format", "choice:html,png"),
    ("plots", "pupils", "Save pupil plots", "bool"),
    ("plots", "pupil_bin", "Pupil plot binning", "int"),
]

PARSERS = {
    "float": float, "int": int, "str": str, "dir": str, "file": str,
    "optfloat": lambda s: float(s) if s.strip() else None,
    "floatlist": lambda s: [float(v) for v in s.split(",") if v.strip()],
}


class FrdGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FRD analysis")
        self.vars, self.frames = {}, {}
        self.log_queue = queue.Queue()
        self._build()
        if DEFAULT_CFG.exists():
            self.set_config(frd_pipeline.load_config(DEFAULT_CFG))
        handler = logging.handlers.QueueHandler(self.log_queue)
        logging.getLogger("frd").addHandler(handler)
        self.after(200, self._poll_log)

    # ---------------------------------------------------------------- layout
    def _build(self):
        form = ttk.Frame(self, padding=6)
        form.grid(row=0, column=0, sticky="nsew")
        for sec, key, label, kind in FIELDS:
            if sec not in self.frames:
                self.frames[sec] = ttk.LabelFrame(form, text=sec, padding=4)
                self.frames[sec].pack(fill="x", pady=2)
                self.frames[sec].columnconfigure(1, weight=1)
            fr = self.frames[sec]
            row = fr.grid_size()[1]
            ttk.Label(fr, text=label).grid(row=row, column=0, sticky="w")
            if kind == "bool":
                var = tk.BooleanVar()
                ttk.Checkbutton(fr, variable=var).grid(row=row, column=1, sticky="w")
            elif kind.startswith("choice:"):
                var = tk.StringVar()
                ttk.Combobox(fr, textvariable=var, values=kind[7:].split(","),
                             state="readonly", width=10).grid(row=row, column=1, sticky="w")
            else:
                var = tk.StringVar()
                ttk.Entry(fr, textvariable=var, width=50).grid(row=row, column=1, sticky="ew")
                if kind in ("dir", "file"):
                    ttk.Button(fr, text="...", width=3,
                               command=lambda v=var, k=kind: self._browse(v, k)
                               ).grid(row=row, column=2)
            self.vars[(sec, key)] = (var, kind)

        bar = ttk.Frame(form)
        bar.pack(fill="x", pady=4)
        ttk.Button(bar, text="Load config", command=self._load).pack(side="left")
        ttk.Button(bar, text="Save config", command=self._save).pack(side="left", padx=4)
        self.run_btn = ttk.Button(bar, text="RUN", command=self._run)
        self.run_btn.pack(side="right")

        self.log_text = tk.Text(self, height=12, width=90, state="disabled")
        self.log_text.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 6))
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

    def _browse(self, var, kind):
        path = filedialog.askdirectory() if kind == "dir" else filedialog.askopenfilename(
            filetypes=[("FITS", "*.fits *.fit *.fts"), ("All", "*")])
        if path:
            var.set(path)

    # ---------------------------------------------------------------- config
    def get_config(self) -> dict:
        cfg = {}
        for (sec, key), (var, kind) in self.vars.items():
            raw = var.get()
            parse = PARSERS.get(kind, lambda v: v)  # bool/choice already typed
            try:
                cfg.setdefault(sec, {})[key] = parse(raw) if isinstance(raw, str) else raw
            except ValueError:
                raise ValueError(f"Invalid value for {sec}.{key}: '{raw}'")
        return cfg

    def set_config(self, cfg: dict):
        for (sec, key), (var, kind) in self.vars.items():
            val = cfg.get(sec, {}).get(key)
            if kind == "bool":
                var.set(bool(val))
            elif kind == "floatlist":
                var.set(", ".join(f"{v:g}" for v in (val or [])))
            else:
                var.set("" if val is None else str(val))

    def _load(self):
        path = filedialog.askopenfilename(filetypes=[("YAML", "*.yaml *.yml")])
        if path:
            self.set_config(frd_pipeline.load_config(path))

    def _save(self):
        path = filedialog.asksaveasfilename(defaultextension=".yaml",
                                            filetypes=[("YAML", "*.yaml *.yml")])
        if path:
            try:
                Path(path).write_text(yaml.safe_dump(self.get_config(), sort_keys=False))
            except ValueError as exc:
                messagebox.showerror("Config error", str(exc))

    # ---------------------------------------------------------------- run
    def _run(self):
        try:
            cfg = self.get_config()
        except ValueError as exc:
            messagebox.showerror("Config error", str(exc))
            return
        for key in ("input_dir", "output_dir"):
            if not cfg["paths"][key]:
                messagebox.showerror("Config error", f"paths.{key} is empty")
                return
        self.run_btn.config(state="disabled")
        threading.Thread(target=self._worker, args=(cfg,), daemon=True).start()

    def _worker(self, cfg):
        log = logging.getLogger("frd")
        try:
            frd_pipeline.run(cfg)
        except Exception as exc:  # report any failure in the GUI log
            log.exception("Analysis failed: %s", exc)
        finally:
            self.log_queue.put(None)  # end-of-run sentinel (Tk calls stay in the main thread)

    def _poll_log(self):
        while not self.log_queue.empty():
            rec = self.log_queue.get_nowait()
            if rec is None:
                self.run_btn.config(state="normal")
                continue
            self.log_text.config(state="normal")
            self.log_text.insert("end", f"{rec.levelname} {rec.getMessage()}\n")
            self.log_text.see("end")
            self.log_text.config(state="disabled")
        self.after(200, self._poll_log)


if __name__ == "__main__":
    FrdGui().mainloop()
