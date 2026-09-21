# Copyright (c) 2026 Alessio Turchi (INAF - Osservatorio Astrofisico di Arcetri)
# Based on the FRD IDL package and its Python port (frd_toolkit.py),
# Copyright (c) Andrea Tozzi (INAF - Osservatorio Astrofisico di Arcetri)
# SPDX-License-Identifier: GPL-3.0-or-later (see LICENSE)

"""FITS discovery, loading and distance assignment for the FRD pipeline."""

import logging
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits

log = logging.getLogger("frd.io")  # child of the pipeline logger


def natural_key(name: str) -> list:
    """Sort key splitting digit runs, so that 'f2' < 'f10'."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def find_fits(input_dir: str, pattern: str) -> List[Path]:
    """Return the matching files of input_dir (non-recursive), natural-sorted by name."""
    files = [p for p in Path(input_dir).glob(pattern) if p.is_file()]
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {input_dir}")
    return sorted(files, key=lambda p: natural_key(p.name))


def load_frame(path: Path) -> np.ndarray:
    """Load the first HDU with data as a 2D float image.

    Leading axes are averaged, so (1, ny, nx) acquisition files and
    multi-frame cubes both become (ny, nx).
    """
    data = np.asarray(fits.getdata(path), dtype=float)
    while data.ndim > 2:
        data = data.mean(axis=0)
    if data.ndim != 2:
        raise ValueError(f"{path.name}: expected a 2D image, got shape {data.shape}")
    return data


def header_distance(path: Path, key: str) -> Optional[float]:
    """Distance stored in the header under `key`, or None if absent/empty/invalid."""
    value = fits.getheader(path).get(key)
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        log.warning("%s: %s='%s' is not a number, ignored", path.name, key, value)
        return None


def assign_distances(
    files: Sequence[Path], use_header: bool, key: str, step_mm: float, offset_mm: float
) -> Tuple[List[Path], np.ndarray, str]:
    """Return (ordered files, distances in mm, description of the distance source).

    Header distances are used only if present in ALL files (files are then
    sorted by distance); otherwise files keep the name order with a fixed step.
    """
    dist = [header_distance(f, key) for f in files] if use_header else [None] * len(files)
    n_found = sum(d is not None for d in dist)

    if use_header and n_found == len(files):
        order = np.argsort(dist, kind="stable")
        return [files[i] for i in order], np.asarray(dist)[order] + offset_mm, f"header {key}"

    if use_header and n_found > 0:
        log.warning("%s found in %d/%d files only: using filename order", key, n_found, len(files))
    dist_mm = np.arange(len(files)) * step_mm + offset_mm
    return list(files), dist_mm, f"filename order, step {step_mm} mm"


class LazyFrames:
    """Read-only sequence that loads (and dark-subtracts) one frame at a time,
    so that long series never sit in memory all together."""

    def __init__(self, files: Sequence[Path], dark: Optional[np.ndarray] = None):
        self.files = list(files)
        self.dark = dark

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> np.ndarray:
        if not -len(self) <= i < len(self):
            raise IndexError(i)
        img = load_frame(self.files[i])
        if self.dark is not None:
            if self.dark.shape != img.shape:
                raise ValueError(f"Dark shape {self.dark.shape} != {self.files[i].name} {img.shape}")
            img = img - self.dark
        return img
