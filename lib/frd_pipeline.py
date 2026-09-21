# Copyright (c) 2026 Alessio Turchi (INAF - Osservatorio Astrofisico di Arcetri)
# Based on the FRD IDL package and its Python port (frd_toolkit.py),
# Copyright (c) Andrea Tozzi (INAF - Osservatorio Astrofisico di Arcetri)
# SPDX-License-Identifier: GPL-3.0-or-later (see LICENSE)

"""FRD analysis pipeline driven by a YAML configuration, using frd_toolkit as library.

Usage (from the frd_analysis/ root): python -m lib.frd_pipeline frd_config.yaml
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

from . import frd_toolkit as tk
from .frd_io import LazyFrames, assign_distances, find_fits, load_frame

log = logging.getLogger("frd")


def load_config(path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def f_at_ee(curve: tk.FrdCurve, levels) -> np.ndarray:
    """Output F/# at which the normalised enclosed energy reaches each level."""
    ee = np.maximum.accumulate(curve.normalized())  # enforce monotonic growth with aperture
    return np.interp(levels, ee, curve.f_number, left=np.nan, right=np.nan)


def save_fig(fig, path_noext: Path, fmt: str) -> None:
    """Save a plotly figure as PNG (needs kaleido) or HTML, falling back to HTML."""
    if fmt == "png":
        try:
            fig.write_image(f"{path_noext}.png")
            return
        except Exception as exc:  # kaleido missing or broken
            log.warning("PNG export failed (%s): saving HTML instead", exc)
    fig.write_html(f"{path_noext}.html", include_plotlyjs="cdn")


def pupil_figure(img, radius, cx, cy, b, title):
    """Pupil plot on a b x b binned image (coordinates rescaled accordingly)."""
    ny, nx = (s // b * b for s in img.shape)
    small = img[:ny, :nx].reshape(ny // b, b, nx // b, b).mean(axis=(1, 3))
    res = tk.PupilResult(radius / b, (cx - (b - 1) / 2) / b, (cy - (b - 1) / 2) / b,
                         np.nan, np.nan, 0)
    return tk.plot_pupil_image(small, res, title=f"{title} (bin {b}x{b})")


def measure_pupils(frames, dist, cfg):
    """Radii/centres of all frames, plus the F/# series fit when >= 3 frames."""
    pc, pix = cfg["pupil"], cfg["detector"]["pix_size_mm"]
    if len(frames) >= 3:
        fres = tk.measure_f_number_series(frames, pix, distances_mm=dist,
                                          sogl=pc["sogl"], s2_p2v=pc["s2_p2v"])
        return fres.radii_pix, fres.centers_x_pix, fres.centers_y_pix, fres
    log.warning("Only %d frame(s): F/# series fit skipped", len(frames))
    res = [tk.analyze_pupil(f, sogl=pc["sogl"], s2_p2v=pc["s2_p2v"])[0] for f in frames]
    return (np.array([r.radius for r in res]), np.array([r.center_x for r in res]),
            np.array([r.center_y for r in res]), None)


def write_summary(path, fres, dist, radii, source, n):
    lines = [f"n_images          {n}", f"distance_source   {source}"]
    if fres is not None:
        fit = stats.linregress(dist, radii)
        lines += [
            f"f_number          {fres.f_number:.4f} +/- {fres.f_number_error:.4f}",
            f"tilt_x_arcmin     {fres.tilt_x_arcmin:.3f} +/- {fres.tilt_x_error_arcmin:.3f}",
            f"tilt_y_arcmin     {fres.tilt_y_arcmin:.3f} +/- {fres.tilt_y_error_arcmin:.3f}",
            # ~0 if the distances are absolute fibre-sensor distances (core size neglected)
            f"r0_distance_mm    {-fit.intercept / fit.slope:.3f}",
        ]
    Path(path).write_text("\n".join(lines) + "\n")
    log.info("Summary:\n  %s", "\n  ".join(lines))


def run(cfg: dict) -> Path:
    """Run the full analysis; return the output directory."""
    p, d, fc, pl = cfg["paths"], cfg["distances"], cfg["frd_curve"], cfg["plots"]
    pix = cfg["detector"]["pix_size_mm"]
    outdir = Path(p["output_dir"]) / datetime.now().strftime("frd_%Y-%m-%d_%H-%M-%S")
    (outdir / "curves").mkdir(parents=True)
    handler = logging.FileHandler(outdir / "frd_analysis.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    try:
        (outdir / "frd_config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        files, dist, source = assign_distances(find_fits(p["input_dir"], p["pattern"]),
                                               d["use_header"], d["header_key"],
                                               d["step_mm"], d["offset_mm"])
        log.info("%d files, distances from %s", len(files), source)
        dark = load_frame(Path(p["dark_file"])) if p.get("dark_file") else None
        frames = LazyFrames(files, dark)

        radii, cx, cy, fres = measure_pupils(frames, dist, cfg)
        with open(outdir / "frd_pupils.txt", "w") as fh:
            fh.write("# index file distance_mm radius_pix diameter_pix center_x center_y\n")
            for i, f in enumerate(files):
                fh.write(f"{i} {f.name} {dist[i]:.4f} {radii[i]:.3f} {2 * radii[i]:.3f} "
                         f"{cx[i]:.3f} {cy[i]:.3f}\n")
        write_summary(outdir / "frd_fnumber.txt", fres, dist, radii, source, len(files))

        curves = []
        if np.any(dist <= 0):
            log.error("Non-positive distances: FRD curves need the absolute fibre-sensor "
                      "distance, set distances.offset_mm. Curves skipped.")
        levels = np.asarray(fc["ee_levels"], float)
        for i, f in enumerate(files):
            img = frames[i]
            if dist[i] > 0:
                curve = tk.compute_frd_curve(img, pix, dist[i], center=(cx[i], cy[i]),
                                             mode=fc["mode"], fn_max=fc["fn_max"],
                                             n_points=fc["n_points"], border_pix=fc["border_pix"])
                np.savetxt(outdir / "curves" / f"{f.stem}_frd.txt",
                           np.column_stack([curve.f_number, curve.diameter_pix,
                                            curve.intensity, curve.normalized()]),
                           fmt="%.6g", header=f"L={dist[i]:.4f} mm\n"
                           "f_number diameter_pix intensity intensity_norm")
                curves.append((f, curve, f_at_ee(curve, levels), dist[i]))
            if pl["pupils"] and tk.go is not None:
                save_fig(pupil_figure(img, radii[i], cx[i], cy[i], max(1, pl["pupil_bin"]), f.name),
                         outdir / "curves" / f"{f.stem}_pupil", pl["format"])

        if curves:
            with open(outdir / "frd_ee_levels.txt", "w") as fh:
                fh.write("# file distance_mm " + " ".join(f"F_EE{lv:g}" for lv in levels) + "\n")
                for f, _, fee, dmm in curves:
                    fh.write(f"{f.name} {dmm:.4f} " + " ".join(f"{v:.4f}" for v in fee) + "\n")

        if tk.go is None:
            log.warning("plotly not installed: plots skipped")
        else:
            if fres is not None:
                save_fig(tk.plot_f_number_fit(fres), outdir / "frd_fnumber_fit", pl["format"])
                save_fig(tk.plot_centroid_migration(fres), outdir / "frd_centroid", pl["format"])
            batch = [(tk.FrdMeasurement(label=f.name, image=np.empty((0, 0))), c)
                     for f, c, _, _ in curves]
            if batch:
                save_fig(tk.plot_frd_batch(batch, normalized=True), outdir / "frd_curves_norm",
                         pl["format"])
                save_fig(tk.plot_frd_batch(batch, normalized=False), outdir / "frd_curves_raw",
                         pl["format"])
        log.info("Done: %s", outdir)
        return outdir
    finally:
        log.removeHandler(handler)
        handler.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(load_config(sys.argv[1]))
