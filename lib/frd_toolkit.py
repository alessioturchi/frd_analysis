# Original IDL code (2008-2012) and Python port: Copyright (c) Andrea Tozzi (INAF - Osservatorio Astrofisico di Arcetri)
# Modifications: Copyright (c) 2026 Alessio Turchi (INAF - Osservatorio Astrofisico di Arcetri)
# SPDX-License-Identifier: GPL-3.0-or-later (see LICENSE)

"""
frd_toolkit.py
================================================================================
Porting Python del pacchetto IDL per la misura della Focal Ratio Degradation
(FRD) e per l'allineamento ottico basato su pupille circolari.

Fonte: procedure .pro originali (A. Tozzi, INAF-Arcetri, ca. 2008-2012).

Corrispondenza con i file .pro originali
-----------------------------------------------------------------------------
  soglia.pro                    -> apply_threshold()
  calcola_baricentro.pro        -> compute_centroid()
  calcola_raggio.pro            -> compute_radius()
  sottosoglia.pro                -> apply_sottosoglia()
  analizza.pro / analizza_cont   -> analyze_pupil()
  make_spot.pro (+ make_mask     -> make_spot()
    mancante nello zip originale,
    reimplementata qui)
  distanza_centri.pro            -> quadrant_distances()
  effenumero.pro                 -> measure_f_number_series()
  allinea.pro                    -> measure_f_number_series() (stessa logica,
                                     allinea.pro era una copia con output
                                     leggermente diverso)
  frd_analysis.pro               -> compute_frd_curve()
  frd_analysis_file.pro          -> batch_frd_analysis()  (la parte di
                                     dataset hardcoded del 2011 è stata
                                     sostituita da una lista di misure
                                     passata dal chiamante, vedi FrdMeasurement)
  acquisition_frames.pro,
  frd_acquisition_frames.pro,
  muovileggirotatore.pro,
  muovileggirotatore2.pro        -> acquire_with_dark(),
                                     acquire_alignment_series()
  move_rotator.pro,
  MuoviLeggiRotatore.pro         -> classi astratte Camera / Rotator

Cosa NON è stato portato
-----------------------------------------------------------------------------
  - I driver hardware specifici (Apogee via Maxim DL + apogee_server.py,
    Thorlabs via CMU1394, rotatore Micos DT65 su seriale, AGW/rerotator
    daemon) non esistono più come tali: qui sono sostituiti da interfacce
    astratte (Camera, Rotator) da implementare per l'hardware che usi oggi.
  - Il link DDE a Zemax (DDE_zemax.pro) è tecnologia Windows/IDL-specifica
    ormai obsoleta; se ti serve automatizzare Zemax oggi conviene usare la
    ZOS-API (pyzdde è deprecato, Zemax OpticStudio espone una API COM/.NET
    utilizzabile da Python) — dimmi se vuoi che la aggiunga.
  - I file duplicati/vecchi (Copia di ANALIZZA.PRO, allinea_rirot_old.pro,
    frd_analysis_file_old.pro, la versione "vecchia" dentro _new_soglia.pro)
    non sono stati portati: la logica corrente è quella riportata sopra.
  - a3d_porosity_cdf.pro non riguarda l'FRD e non è stato incluso.

Dipendenze: numpy, scipy (fit lineare con errore), astropy (I/O FITS),
pillow (TIFF/BMP), plotly (grafici, opzionale ma consigliata).
================================================================================
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

try:
    from astropy.io import fits
except ImportError:  # pragma: no cover
    fits = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover
    go = None


# ==============================================================================
# 1. NUCLEO DI CALCOLO: sogliatura, baricentro, raggio, analisi pupilla
#    (soglia.pro, calcola_baricentro.pro, calcola_raggio.pro, sottosoglia.pro,
#     analizza.pro)
# ==============================================================================
#
# Convenzione usata in questo modulo: le immagini sono array numpy 2D con
# shape (ny, nx) (indicizzazione [riga, colonna], come al solito in numpy).
# I "centri" sono invece riportati come coppie (x, y) in pixel, coerentemente
# con l'uso IDL originale (centro[0]=x, centro[1]=y).


def apply_threshold(image: np.ndarray, s: float) -> np.ndarray:
    """Port di soglia.pro: azzera tutti i pixel sotto la soglia s.

    Ritorna una NUOVA immagine (non modifica l'originale), a differenza della
    versione IDL che operava in place — è più sicuro in Python dato che gli
    array numpy sono mutabili per riferimento.
    """
    out = image.copy()
    out[out < s] = 0
    return out


def compute_centroid(thresholded_image: np.ndarray) -> np.ndarray:
    """Port di calcola_baricentro.pro: baricentro GEOMETRICO (non pesato
    sull'intensità) dei pixel > 0, cioè il centro dell'area sogliata.

    NOTA: la routine IDL costruiva la maschera binaria dei pixel > 0 e ne
    calcolava il centro come media (non pesata) delle coordinate x e y —
    esattamente equivalente a quanto fatto qui con np.nonzero(), solo
    espresso senza il prodotto matriciale IDL (##) che serviva solo a
    generare le griglie di indici.

    Ritorna [x_b, y_b].
    """
    ys, xs = np.nonzero(thresholded_image > 0)
    if xs.size == 0:
        raise ValueError("Nessun pixel sopra soglia: baricentro non definito")
    return np.array([xs.mean(), ys.mean()])


def compute_radius(thresholded_image: np.ndarray, s: float) -> float:
    """Port di calcola_raggio.pro: raggio equivalente assumendo come area il
    numero di pixel sopra/uguali alla soglia s (area = n_pixel, r = sqrt(area/pi)).
    """
    n_pixels = np.count_nonzero(thresholded_image >= s)
    area = n_pixels / np.pi
    return float(np.sqrt(area))


def apply_sottosoglia(
    image: np.ndarray,
    s: float,
    r: float,
    center: Sequence[float],
    edge_margin: float = 0.1,
    edge_thickness: float = 10.0,
) -> Tuple[int, np.ndarray]:
    """Port di sottosoglia.pro.

    Riempie con il valore di soglia `s` tutti i pixel interni al cerchio di
    raggio (r - edge_margin) centrato in `center`, così che eventuali "buchi"
    di rumore sotto soglia dentro il cerchio trovato non vengano rimossi
    dalla sogliatura. Questo è il passo che rende iterativo (e convergente)
    analyze_pupil(): ad ogni iterazione il cerchio individuato viene
    "bloccato" a valore s finché il centro/raggio ricalcolato non cambia più.

    Modifica `image` IN PLACE (come la versione IDL) perché è così che viene
    usata dentro il ciclo di analyze_pupil().

    Ritorna (pixel, edge_indices):
      pixel        = numero di pixel interni al cerchio se sono stati
                      modificati, 0 se erano già tutti al valore di soglia
                      (condizione di convergenza del ciclo chiamante)
      edge_indices = indici (riga, colonna) dei pixel sull'anello di raggio r
                      (usati in IDL solo per l'overlay grafico live)
    """
    ny, nx = image.shape
    x0, y0 = center
    yy, xx = np.indices((ny, nx))
    d2 = (xx - x0) ** 2 + (yy - y0) ** 2

    inside = d2 < (r - edge_margin) ** 2
    n_inside = int(np.count_nonzero(inside))

    if n_inside > 0 and not np.all(image[inside] == s):
        image[inside] = s
        pixel = n_inside
    else:
        pixel = 0

    edge_mask = np.abs(d2 - r**2) < edge_thickness
    edge_indices = np.nonzero(edge_mask)

    return pixel, edge_indices


@dataclass
class PupilResult:
    """Risultato dell'analisi di UNA pupilla (una riga di 'results'/'m' nel
    codice IDL originale)."""

    radius: float
    center_x: float
    center_y: float
    threshold1: float
    threshold2: float
    n_iterations: int

    @property
    def diameter(self) -> float:
        return 2.0 * self.radius

    @property
    def center(self) -> Tuple[float, float]:
        return (self.center_x, self.center_y)


def _analyze_single_pupil(
    image: np.ndarray,
    s1_fraction: Optional[float] = None,
    s2_fraction: Optional[float] = None,
    max_iter: int = 200,
) -> PupilResult:
    """Nucleo iterativo comune a un singolo riquadro immagine (usato sia per
    il caso singola pupilla che per ciascuno dei 4 quadranti in modalità
    SPLIT). Corrisponde al corpo del ciclo `for pup=0,n_pup-1` di analizza.pro.
    """
    imm = image.astype(float)

    # PRIMA SOGLIA: percentuale sull'escursione picco-valle dell'immagine
    if s1_fraction is None:
        s1_fraction = 0.05
    s1 = imm.min() + (imm.max() - imm.min()) * s1_fraction
    imm_soglia = apply_threshold(imm, s1)

    # SECONDA SOGLIA: se non data una frazione esplicita, e' la media dei
    # valori sopra la prima soglia (comportamento di default in analizza.pro)
    if s2_fraction is not None:
        s2 = (imm_soglia.max() - imm_soglia.min()) * s2_fraction
    else:
        above = imm[imm > s1]
        if above.size == 0:
            raise ValueError("Nessun pixel sopra la prima soglia")
        s2 = float(above.mean())

    imm_soglia = apply_threshold(imm_soglia, s2)

    n_iter = 0
    while True:
        n_iter += 1
        centro = compute_centroid(imm_soglia)
        r = compute_radius(imm_soglia, s2)
        pixel, _edge = apply_sottosoglia(imm_soglia, s2, r, centro)
        if pixel == 0 or n_iter >= max_iter:
            break

    return PupilResult(
        radius=r,
        center_x=float(centro[0]),
        center_y=float(centro[1]),
        threshold1=s1,
        threshold2=s2,
        n_iterations=n_iter,
    )


def analyze_pupil(
    image: np.ndarray,
    split: bool = False,
    sogl: Optional[float] = None,
    s2_p2v: Optional[float] = None,
    x_shift: float = 50.0,
    y_shift: float = 10.0,
) -> List[PupilResult]:
    """Port di analizza.pro (e del quasi-duplicato analizza_cont.pro).

    Trova centro e raggio della/e pupilla/e in un'immagine tramite
    sogliatura iterativa a "cerchio che si stabilizza" (vedi apply_sottosoglia).

    Parametri
    ---------
    image    : immagine 2D (ny, nx)
    split    : se True, divide l'immagine in 4 quadranti (usata per le
               piastre di allineamento a 4 fori) e analizza ciascuno,
               ricentrando poi le coordinate nel sistema dell'immagine
               originale
    sogl     : frazione (0-1) picco-valle per la prima soglia (default 0.05
               come in analizza.pro; NB. il default usato da allinea.pro e
               dalla maggior parte delle chiamate reali nel codice originale
               era 0.3 — passalo esplicitamente per replicare quel comportamento)
    s2_p2v   : frazione (0-1) per la seconda soglia; se None, e' la media dei
               pixel sopra la prima soglia (comportamento di default IDL)
    x_shift, y_shift : offset del centro di split rispetto al centro
               geometrico del CCD (stessi parametri hardcoded in analizza.pro)

    Ritorna una lista di PupilResult: 1 elemento se split=False, 4 se
    split=True (ordine: [dx-basso, sx-basso, dx-alto, sx-alto] come nell'IDL
    originale, cioè k[0]=quadrante -x+y, k[1]=+x+y, k[2]=-x-y, k[3]=+x-y).
    """
    ny, nx = image.shape

    if not split:
        result = _analyze_single_pupil(image, sogl, s2_p2v)
        return [result]

    c = int(nx / 2.5)
    d = int(ny / 2.5)
    cx = nx // 2 + int(x_shift)
    cy = ny // 2 + int(y_shift)

    quadrants = [
        image[cy : cy + d, cx - c : cx],  # k[0]
        image[cy : cy + d, cx : cx + c],  # k[1]
        image[cy - d : cy, cx - c : cx],  # k[2]
        image[cy - d : cy, cx : cx + c],  # k[3]
    ]
    offsets = [(0, d), (c, d), (0, 0), (c, 0)]

    results = []
    for quad, (ox, oy) in zip(quadrants, offsets):
        r = _analyze_single_pupil(quad, sogl, s2_p2v)
        r.center_x += ox
        r.center_y += oy
        results.append(r)

    return results


# ==============================================================================
# 2. MASCHERA CIRCOLARE / SPOT SINTETICO
#    (make_spot.pro — la dipendenza make_mask.pro non era presente nello zip
#     originale ed è stata reimplementata qui direttamente)
# ==============================================================================


def make_spot(
    nx: int,
    ny: int,
    x_center: float,
    y_center: float,
    diameter: float,
    obstruction_diameter: float = 0.0,
) -> np.ndarray:
    """Port (e reimplementazione, vedi nota sopra) di make_spot.pro.

    Genera una maschera circolare (con eventuale ostruzione centrale, per
    pupille ostruite tipo Cassegrain) di diametro `diameter` pixel, centrata
    in (x_center, y_center), dentro un frame (ny, nx).

    Ritorna un array float con 1.0 dentro l'apertura e 0.0 fuori.
    """
    yy, xx = np.indices((ny, nx))
    r2 = (xx - x_center) ** 2 + (yy - y_center) ** 2
    mask = r2 <= (diameter / 2.0) ** 2
    if obstruction_diameter > 0:
        mask &= r2 >= (obstruction_diameter / 2.0) ** 2
    return mask.astype(float)


# ==============================================================================
# 3. DISTANZE FRA CENTRI (piastra di allineamento a 4 fori)
#    (distanza_centri.pro)
# ==============================================================================


@dataclass
class QuadrantGeometry:
    side_lengths: np.ndarray  # 4 lati del quadrato (pix)
    mean_side: float
    diagonal: float
    mean_diameter: float
    diameter_side_ratio: float


def quadrant_distances(pupils: Sequence[PupilResult]) -> QuadrantGeometry:
    """Port di distanza_centri.pro: dati i 4 PupilResult (da analyze_pupil
    con split=True), calcola i lati e la diagonale del quadrato formato dai
    centri, e il rapporto diametro-medio/lato-medio (diagnostica di
    allineamento della piastra a 4 fori).
    """
    if len(pupils) != 4:
        raise ValueError("Servono esattamente 4 PupilResult (analyze_pupil con split=True)")

    centers = np.array([p.center for p in pupils])  # shape (4,2)
    sides = np.zeros(4)

    for i in range(4):
        if i in (1, 3):
            continue  # questi indici sono le diagonali, gestite sotto
        j = (i + 1) % 4
        sides[i] = np.linalg.norm(centers[j] - centers[i])

    for h in range(2):
        k = (h + 2) % 4
        d = np.linalg.norm(centers[k] - centers[h])
        sides[h * 2 + 1] = d

    mean_diameter = float(np.mean([p.diameter for p in pupils]))
    mean_side = float(np.mean(sides))
    diagonal = float(np.sqrt(2) * mean_side)

    return QuadrantGeometry(
        side_lengths=sides,
        mean_side=mean_side,
        diagonal=diagonal,
        mean_diameter=mean_diameter,
        diameter_side_ratio=mean_diameter / mean_side,
    )


# ==============================================================================
# 4. MISURA DEL NUMERO F DA UNA SERIE DI IMMAGINI A DISTANZE NOTE
#    (effenumero.pro, allinea.pro — stessa logica, unificate qui)
# ==============================================================================


@dataclass
class FNumberResult:
    distances_mm: np.ndarray
    radii_pix: np.ndarray
    centers_x_pix: np.ndarray
    centers_y_pix: np.ndarray
    f_number: float
    f_number_error: float
    tilt_x_arcmin: float
    tilt_x_error_arcmin: float
    tilt_y_arcmin: float
    tilt_y_error_arcmin: float


def _linfit_with_error(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    """Fit lineare y = a + b*x con stima dell'errore su b, equivalente a
    LINFIT(...,sigma=...) di IDL (pesi uniformi, come nel codice originale
    che non passava pesi espliciti). Usa scipy.stats.linregress.

    Ritorna (a, b, a_err, b_err) = (intercetta, pendenza, err. intercetta,
    err. pendenza).
    """
    fit = stats.linregress(x, y)
    return fit.intercept, fit.slope, fit.intercept_stderr, fit.stderr


def measure_f_number_series(
    images: Sequence[np.ndarray],
    pix_size_mm: float,
    dist_step_mm: float = 25.0,
    sogl: float = 0.3,
    s2_p2v: float = 0.02,
    distances_mm: Optional[Sequence[float]] = None,
) -> FNumberResult:
    """Port unificato di effenumero.pro e allinea.pro (caso singola pupilla).

    Dato un elenco di immagini acquisite a distanze lineari crescenti e note
    (tipicamente spostando la camera/lo schermo lungo l'asse ottico a passi
    fissi `dist_step_mm`), misura il raggio della pupilla in ciascuna
    immagine e ne ricava il numero F tramite fit lineare raggio-vs-distanza:

        f_number = 1 / (2 * pendenza * pix_size_mm)

    e la deriva (tilt) del centro della pupilla lungo la serie, convertita
    in arcmin.

    Parametri
    ---------
    images        : sequenza di immagini 2D, una per posizione
    pix_size_mm   : dimensione del pixel del sensore (mm)
    dist_step_mm  : passo fra un'acquisizione e la successiva (mm), usato
                    solo se `distances_mm` non è fornito esplicitamente
    sogl, s2_p2v  : soglie passate a analyze_pupil() (i default qui
                    ripropongono quelli usati nelle chiamate reali di
                    allinea.pro, non il default a sé stante di analizza.pro)
    distances_mm  : distanze assolute esplicite, se non equispaziate
    """
    n_im = len(images)
    if distances_mm is None:
        distances_mm = np.arange(n_im) * dist_step_mm
    x = np.asarray(distances_mm, dtype=float)

    radii = np.zeros(n_im)
    xc = np.zeros(n_im)
    yc = np.zeros(n_im)

    for i, img in enumerate(images):
        res = analyze_pupil(img, split=False, sogl=sogl, s2_p2v=s2_p2v)[0]
        radii[i] = res.radius
        xc[i] = res.center_x
        yc[i] = res.center_y

    _, b_r, _, b_r_err = _linfit_with_error(x, radii)
    f_number = 1.0 / (b_r * 2.0 * pix_size_mm)
    f_number_error = abs(f_number * (b_r_err / b_r))

    def _tilt(coord):
        _, b, _, b_err = _linfit_with_error(x, coord)
        rad = b * pix_size_mm
        arcmin = rad * 180.0 * 60.0 / np.pi
        arcmin_err = abs(arcmin * (b_err / b))
        return arcmin, arcmin_err

    tilt_x, tilt_x_err = _tilt(xc)
    tilt_y, tilt_y_err = _tilt(yc)

    return FNumberResult(
        distances_mm=x,
        radii_pix=radii,
        centers_x_pix=xc,
        centers_y_pix=yc,
        f_number=f_number,
        f_number_error=f_number_error,
        tilt_x_arcmin=tilt_x,
        tilt_x_error_arcmin=tilt_x_err,
        tilt_y_arcmin=tilt_y,
        tilt_y_error_arcmin=tilt_y_err,
    )


# ==============================================================================
# 5. CURVA FRD (intensità integrata vs f/# di uscita)
#    (frd_analysis.pro, frd_analysis_file.pro)
# ==============================================================================


@dataclass
class FrdCurve:
    f_number: np.ndarray
    diameter_pix: np.ndarray
    intensity: np.ndarray

    def normalized(self) -> np.ndarray:
        """Intensità normalizzata all'ultimo punto (massima apertura), come
        nel grafico 'Grafico FRD normalizzato' di frd_analysis_file.pro."""
        return self.intensity / self.intensity[-1]


def compute_frd_curve(
    image: np.ndarray,
    pix_size_mm: float,
    dist_fiber_to_ccd_mm: float = 17.5,
    center: Optional[Tuple[float, float]] = None,
    mode: str = "pupil",
    fn_max: float = 10.0,
    n_points: int = 30,
    border_pix: float = 0.0,
    sogl: float = 0.1,
    s2_p2v: float = 0.2,
) -> FrdCurve:
    """Port di frd_analysis.pro (i calcoli /FRD_PUP e /FRD_CCD).

    Integra l'immagine (già mediata e corretta di dark) dentro aperture
    circolari concentriche di diametro crescente, a passi regolari fra il
    diametro minimo (corrispondente a fn_max) e il diametro massimo
    disponibile, per ottenere la curva intensità-vs-f/# di uscita tipica
    di una misura di FRD.

    Parametri
    ---------
    image                : immagine 2D già mediata/dark-sottratta
    pix_size_mm          : dimensione pixel del sensore (mm)
    dist_fiber_to_ccd_mm : distanza capofibra-sensore (mm)
    center               : centro dello spot in pixel (x, y). Se None e
                            mode='pupil', viene calcolato con analyze_pupil()
    mode                 : 'pupil' -> integra a partire dal centro reale
                            della pupilla, come /FRD_PUP
                            'ccd'   -> integra a partire dal centro
                            geometrico del CCD, come /FRD_CCD
    fn_max                : f/# di uscita massimo di analisi (diametro minimo)
    n_points               : numero di punti della curva
    border_pix             : bordo del CCD da escludere dal diametro massimo

    Ritorna FrdCurve(f_number, diameter_pix, intensity).
    """
    ny, nx = image.shape

    if mode == "pupil":
        if center is None:
            res = analyze_pupil(image, sogl=sogl, s2_p2v=s2_p2v)[0]
            center = res.center
        cx, cy = center
        # diametro massimo = doppio della distanza minima dal centro al bordo
        max_diam = 2.0 * min(cx, cy, nx - cx, ny - cy) - border_pix
    elif mode == "ccd":
        cx, cy = nx / 2.0, ny / 2.0
        max_diam = min(nx, ny) - border_pix
    else:
        raise ValueError("mode deve essere 'pupil' o 'ccd'")

    fn_diam_min = (dist_fiber_to_ccd_mm / fn_max) / pix_size_mm
    fn_diam_step = (max_diam - fn_diam_min) / n_points

    diam = np.round((np.arange(1, n_points + 1)) * fn_diam_step + fn_diam_min)
    intensity = np.zeros(n_points)

    for i, d in enumerate(diam):
        aperture = make_spot(nx, ny, cx, cy, d)
        intensity[i] = np.sum(image * aperture)

    f_number = dist_fiber_to_ccd_mm / pix_size_mm / diam

    return FrdCurve(f_number=f_number, diameter_pix=diam, intensity=intensity)


@dataclass
class FrdMeasurement:
    """Descrive UNA misura del dataset batch (sostituisce una riga del CASE
    hardcoded di frd_analysis_file.pro: qui il dataset lo passa il
    chiamante invece di essere scritto nel codice)."""

    label: str
    image: np.ndarray
    dark: Optional[np.ndarray] = None
    input_f_number: Optional[float] = None
    integration_time_s: Optional[float] = None


def batch_frd_analysis(
    measurements: Sequence[FrdMeasurement],
    pix_size_mm: float,
    dist_fiber_to_ccd_mm: float = 17.5,
    mode: str = "pupil",
    qe: float = 0.5,
    adu_per_electron: float = 5.0,
    **frd_kwargs,
) -> List[Tuple[FrdMeasurement, FrdCurve]]:
    """Port generalizzato di frd_analysis_file.pro: applica compute_frd_curve
    a una serie di misure (invece del dataset 2011 hardcoded nel CASE
    originale), sottraendo il dark se presente e convertendo in fotoni
    incidenti se è nota l'integration time (stessa formula
    ADU/QE/tempo_integrazione dell'originale).
    """
    results = []
    for meas in measurements:
        img = meas.image.astype(float)
        if meas.dark is not None:
            img = img - meas.dark

        curve = compute_frd_curve(img, pix_size_mm, dist_fiber_to_ccd_mm, mode=mode, **frd_kwargs)

        if meas.integration_time_s:
            factor = adu_per_electron / qe / meas.integration_time_s
            curve = FrdCurve(
                f_number=curve.f_number,
                diameter_pix=curve.diameter_pix,
                intensity=curve.intensity * factor,
            )

        results.append((meas, curve))

    return results


# ==============================================================================
# 6. INTERFACCE HARDWARE GENERICHE
#    (sostituiscono: read_frames_apogee.pro, read_frames_thorlabs.pro,
#     apogee_act.pro, prepare_my_apogee.pro, move_rotator.pro,
#     MuoviLeggiRotatore.pro — tutte legate a driver/hardware specifici del
#     laboratorio originale e non più utilizzabili come tali)
# ==============================================================================


class Camera(ABC):
    """Interfaccia generica per una camera di acquisizione.

    Implementa questa classe per il tuo hardware attuale (es. una camera
    scientifica pilotata via SDK/driver Python, o anche un semplice wrapper
    su un file system watcher se acquisisci con software esterno).
    """

    @abstractmethod
    def acquire(
        self,
        n_frames: int = 1,
        exposure_time: Optional[float] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        """Acquisisce n_frames. roi = (x1, y1, dx, dy) in pixel, o None per
        il frame intero. Ritorna un array 2D (ny, nx) se n_frames=1, oppure
        3D (n_frames, ny, nx) se n_frames>1 — analogo a read_frames_apogee.pro
        / read_frames_thorlabs.pro."""
        raise NotImplementedError


class Rotator(ABC):
    """Interfaccia generica per un asse motorizzato di rotazione
    (era: rotatore Micos DT65 su seriale in move_rotator.pro, oppure il
    "rerotator" pilotato dal demone AGW in MuoviLeggiRotatore.pro)."""

    @abstractmethod
    def move_to(self, position_deg: float) -> None:
        """Muove l'asse alla posizione assoluta in gradi."""
        raise NotImplementedError

    @abstractmethod
    def home(self) -> None:
        """Riporta l'asse alla posizione di home."""
        raise NotImplementedError

    def get_position(self) -> Optional[float]:
        """Posizione corrente, se leggibile. Opzionale."""
        return None


class SimulatedCamera(Camera):
    """Implementazione di test/sviluppo: genera un blob gaussiano sintetico
    al posto di un frame reale. Utile per validare il resto della pipeline
    senza hardware collegato."""

    def __init__(self, nx: int = 512, ny: int = 512, radius_pix: float = 40.0,
                 peak_counts: float = 1000.0, noise_std: float = 5.0):
        self.nx, self.ny = nx, ny
        self.radius_pix = radius_pix
        self.peak_counts = peak_counts
        self.noise_std = noise_std
        self._rng = np.random.default_rng()

    def acquire(self, n_frames=1, exposure_time=None, roi=None) -> np.ndarray:
        def _one_frame():
            yy, xx = np.indices((self.ny, self.nx))
            cx, cy = self.nx / 2, self.ny / 2
            r2 = (xx - cx) ** 2 + (yy - cy) ** 2
            frame = self.peak_counts * (r2 <= self.radius_pix**2).astype(float)
            frame += self._rng.normal(0, self.noise_std, frame.shape)
            if roi is not None:
                x1, y1, dx, dy = roi
                frame = frame[y1 : y1 + dy, x1 : x1 + dx]
            return frame

        if n_frames == 1:
            return _one_frame()
        return np.stack([_one_frame() for _ in range(n_frames)])


def generate_synthetic_frd_images(
    n_images: int = 5,
    nx: int = 1200,
    ny: int = 1200,
    radius_start_pix: float = 150.0,
    radius_step_pix: float = 30.0,
    center: Optional[Tuple[float, float]] = None,
    peak_counts: float = 1000.0,
    background_counts: float = 0.0,
    noise_fraction: float = 0.10,
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """Genera una serie riproducibile di immagini sintetiche di pupilla.

    Il sigma della gaussiana cresce da un'immagine alla successiva, come nella
    serie usata da ``measure_f_number_series``. I frame ritornati sono immagini
    2D ``float`` pronte per le funzioni di analisi.
    """
    if n_images < 1:
        raise ValueError("n_images deve essere positivo")
    if radius_start_pix <= 0 or radius_step_pix < 0:
        raise ValueError("i sigma devono essere positivi e crescenti")
    if not 0 <= noise_fraction:
        raise ValueError("noise_fraction non puo essere negativo")

    cx, cy = center if center is not None else (nx / 2.0, ny / 2.0)
    yy, xx = np.indices((ny, nx))
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    rng = np.random.default_rng(seed)
    images = []

    for index in range(n_images):
        sigma = radius_start_pix + radius_step_pix * index
        signal = peak_counts * np.exp(-r2 / (2.0 * sigma**2))
        frame = signal + background_counts
        if noise_fraction > 0:
            frame = frame + rng.normal(0, peak_counts * noise_fraction, frame.shape)
        images.append(np.clip(frame, 0.0, None))

    return images


class SimulatedRotator(Rotator):
    """Implementazione di test: tiene traccia della posizione senza
    muovere nulla di reale."""

    def __init__(self):
        self._position = 0.0

    def move_to(self, position_deg: float) -> None:
        self._position = position_deg

    def home(self) -> None:
        self._position = 0.0

    def get_position(self) -> Optional[float]:
        return self._position


# ------------------------------------------------------------------------------
# Routine di acquisizione generiche (usano solo le interfacce sopra)
# ------------------------------------------------------------------------------


def acquire_with_dark(
    camera: Camera,
    n_frames: int,
    exposure_time: Optional[float] = None,
    roi: Optional[Tuple[int, int, int, int]] = None,
    take_dark: bool = False,
    dark_prompt: Callable[[str], str] = input,
) -> Tuple[np.ndarray, Optional[np.ndarray], float, Optional[float]]:
    """Port generalizzato di acquisition_frames.pro / frd_acquisition_frames.pro.

    Acquisisce n_frames (con la Camera fornita), opzionalmente un frame di
    dark (con conferma interattiva, come "switch off the source and press
    return" nell'originale), e ne calcola il totale.

    Ritorna (frames, dark_frame_o_None, total_counts, total_counts_dark_o_None).
    """
    frames = camera.acquire(n_frames, exposure_time=exposure_time, roi=roi)
    total = float(np.sum(frames)) / n_frames

    dark_frame = None
    total_dark = None
    if take_dark:
        dark_prompt("Dark frame (switch off the source and press return)")
        dark_frame = camera.acquire(1, exposure_time=exposure_time, roi=roi)
        total_dark = float(np.sum(dark_frame))

    return frames, dark_frame, total, total_dark


def acquire_alignment_series(
    camera: Camera,
    n_steps: int,
    step_deg: float,
    rotator: Optional[Rotator] = None,
    exposure_time: Optional[float] = None,
    roi: Optional[Tuple[int, int, int, int]] = None,
    save_dir: Optional[Path] = None,
    prefix: str = "frame",
    settle_time_s: float = 0.0,
) -> List[np.ndarray]:
    """Port generalizzato di muovileggirotatore.pro / muovileggirotatore2.pro:
    muove il rotatore a passi fissi e acquisisce un frame ad ogni posizione,
    salvando opzionalmente su disco in FITS. A differenza dell'originale, che
    acquisiva in parallelo da due camere diverse (Apogee + Thorlabs), qui la
    serie è generica su una singola Camera: per replicare l'acquisizione
    doppia basta chiamare la funzione due volte con due Camera diverse, o
    farne un wrapper che le combina in acquire().
    """
    frames = []
    for i in range(n_steps):
        if rotator is not None:
            rotator.move_to(i * step_deg)
            if settle_time_s:
                time.sleep(settle_time_s)

        frame = camera.acquire(1, exposure_time=exposure_time, roi=roi)
        frames.append(frame)

        if save_dir is not None and fits is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            fits.writeto(save_dir / f"{prefix}_{i}.fits", frame.astype(np.float32), overwrite=True)

    if rotator is not None:
        rotator.home()

    return frames


# ==============================================================================
# 7. I/O IMMAGINI (equivalenti a readfits/read_tiff/read_bmp usati nell'IDL)
# ==============================================================================


def load_image(path: Path) -> np.ndarray:
    """Carica un'immagine FITS o TIFF/BMP/PNG restituendo un array float 2D
    (o 3D se il FITS contiene un cubo di frame, come in frd_analysis.pro)."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".fit", ".fits", ".fts"):
        if fits is None:
            raise ImportError("astropy non installato: 'pip install astropy' per leggere FITS")
        with fits.open(path) as hdul:
            return hdul[0].data.astype(float)

    if Image is None:
        raise ImportError("Pillow non installato: 'pip install pillow' per leggere TIFF/BMP/PNG")
    return np.asarray(Image.open(path)).astype(float)


# ==============================================================================
# 8. VISUALIZZAZIONE (Plotly)
#    Sostituiscono le window IDL (tvscl, plot, oplot) di analizza.pro,
#    allinea.pro/effenumero.pro e frd_analysis.pro/frd_analysis_file.pro.
#
#    Tutte le funzioni ritornano un go.Figure senza chiamare .show(): puoi
#    quindi visualizzarla (fig.show()), salvarla (fig.write_html(...)) o
#    comporla in una dashboard.
# ==============================================================================


def _require_plotly():
    if go is None:
        raise ImportError("plotly non installato: 'pip install plotly'")


def plot_pupil_image(
    image: np.ndarray,
    result: Optional[PupilResult] = None,
    title: str = "Pupilla",
) -> "go.Figure":
    """Mostra l'immagine (heatmap) con overlay del cerchio trovato da
    analyze_pupil(), equivalente al tvscl + overlay 'cerchio' di analizza.pro."""
    _require_plotly()
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=image, colorscale="Viridis", showscale=True))

    if result is not None:
        theta = np.linspace(0, 2 * np.pi, 200)
        cx, cy = result.center
        fig.add_trace(
            go.Scatter(
                x=cx + result.radius * np.cos(theta),
                y=cy + result.radius * np.sin(theta),
                mode="lines",
                line=dict(color="red", width=2),
                name=f"r={result.radius:.1f} px",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[cx], y=[cy], mode="markers",
                marker=dict(color="red", symbol="cross", size=10),
                name="centro",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="x [pix]",
        yaxis_title="y [pix]",
        yaxis=dict(scaleanchor="x"),
        width=600, height=600,
    )
    return fig


def plot_f_number_fit(result: FNumberResult, title: str = "Misura F/#") -> "go.Figure":
    """Diametro vs distanza + retta di fit, equivalente a window,1 di
    allinea.pro/effenumero.pro."""
    _require_plotly()
    x = result.distances_mm
    # ricostruisco la retta di fit dai dati (diametro = 2*raggio) solo per
    # disegnarla; i valori ufficiali di F/# e relativo errore restano quelli
    # calcolati da measure_f_number_series() sul raggio (non sul diametro)
    diam = 2 * result.radii_pix
    b, a = np.polyfit(x, diam, 1)  # solo per disegnare la retta, i valori
    # ufficiali (con errore) restano quelli di measure_f_number_series()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=diam, mode="markers", name="dati", marker=dict(size=9)))
    fig.add_trace(go.Scatter(x=x, y=a + b * x, mode="lines", name="fit lineare"))
    fig.update_layout(
        title=f"{title}<br><sup>F/# = {result.f_number:.3f} ± {result.f_number_error:.3f}</sup>",
        xaxis_title="distanza [mm]",
        yaxis_title="diametro pupilla [pix]",
        width=700, height=450,
    )
    return fig


def plot_centroid_migration(result: FNumberResult, title: str = "Migrazione del centro") -> "go.Figure":
    """Traiettoria del centro della pupilla lungo la serie di distanze,
    equivalente a window,4 di allinea.pro."""
    _require_plotly()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.centers_x_pix, y=result.centers_y_pix,
            mode="markers+lines", marker=dict(size=9),
            text=[f"{d:.1f} mm" for d in result.distances_mm],
            hovertemplate="x=%{x:.2f} y=%{y:.2f} (%{text})",
        )
    )
    fig.update_layout(
        title=(
            f"{title}<br><sup>tilt x = {result.tilt_x_arcmin:.2f} ± "
            f"{result.tilt_x_error_arcmin:.2f} arcmin, "
            f"tilt y = {result.tilt_y_arcmin:.2f} ± "
            f"{result.tilt_y_error_arcmin:.2f} arcmin</sup>"
        ),
        xaxis_title="x centro [pix]",
        yaxis_title="y centro [pix]",
        yaxis=dict(scaleanchor="x"),
        width=600, height=600,
    )
    return fig


def plot_frd_curve(
    curve: FrdCurve,
    normalized: bool = False,
    title: str = "Curva FRD",
    figure: Optional["go.Figure"] = None,
    name: Optional[str] = None,
) -> "go.Figure":
    """Intensità (o intensità normalizzata) vs f/# di uscita, equivalente ai
    plot di frd_analysis.pro e ai grafici sovrapposti/normalizzati di
    frd_analysis_file.pro (window 5/7/8).

    Se `figure` è fornita, la traccia viene aggiunta a quella figura invece
    di crearne una nuova — utile per sovrapporre più curve (una per fibra o
    per configurazione), come nell'originale IDL.
    """
    _require_plotly()
    fig = figure if figure is not None else go.Figure()
    y = curve.normalized() if normalized else curve.intensity
    fig.add_trace(
        go.Scatter(
            x=curve.f_number, y=y, mode="markers+lines",
            name=name or "FRD",
        )
    )
    if figure is None:
        fig.update_layout(
            title=title,
            xaxis_title="F/# di uscita",
            yaxis_title="Intensità normalizzata" if normalized else "Intensità [counts]",
            width=750, height=500,
        )
    return fig


def plot_frd_image_and_curve(
    image: np.ndarray,
    curve: FrdCurve,
    normalized: bool = True,
    title: str = "Analisi FRD",
) -> "go.Figure":
    """Mostra affiancate l'immagine di input e la curva FRD calcolata."""
    _require_plotly()
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Immagine grezza", "Curva FRD"),
        horizontal_spacing=0.10,
    )
    fig.add_trace(
        go.Heatmap(z=image, colorscale="Viridis", showscale=True),
        row=1,
        col=1,
    )
    y = curve.normalized() if normalized else curve.intensity
    fig.add_trace(
        go.Scatter(
            x=curve.f_number,
            y=y,
            mode="markers+lines",
            name="FRD",
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="x [pix]", row=1, col=1)
    fig.update_yaxes(title_text="y [pix]", row=1, col=1)
    fig.update_xaxes(title_text="F/# di uscita", row=1, col=2)
    fig.update_yaxes(
        title_text="Intensità normalizzata" if normalized else "Intensità [counts]",
        row=1,
        col=2,
    )
    fig.update_layout(title=title, width=1200, height=600)
    return fig


def plot_frd_batch(
    results: Sequence[Tuple[FrdMeasurement, FrdCurve]],
    normalized: bool = True,
    title: str = "FRD - confronto misure",
) -> "go.Figure":
    """Sovrappone le curve FRD di più misure (output di batch_frd_analysis),
    equivalente ai grafici a più tracce di frd_analysis_file.pro (window 7/8)."""
    _require_plotly()
    fig = go.Figure()
    for meas, curve in results:
        plot_frd_curve(curve, normalized=normalized, figure=fig, name=meas.label)
    fig.update_layout(
        title=title,
        xaxis_title="F/# di uscita",
        yaxis_title="Intensità normalizzata" if normalized else "Intensità [counts]",
        width=800, height=500,
    )
    return fig


# ==============================================================================
# Auto-test rapido su dati sintetici (eseguito solo se il file viene lanciato
# direttamente, non quando viene importato come modulo)
# ==============================================================================

if __name__ == "__main__":
    print("Self-test su dati sintetici...")

    # --- test analyze_pupil su un disco netto ---
    cam = SimulatedCamera(nx=400, ny=400, radius_pix=50, peak_counts=1000, noise_std=3)
    frame = cam.acquire()
    res = analyze_pupil(frame, sogl=0.3, s2_p2v=0.2)[0]
    print(f"  analyze_pupil: raggio atteso=50.0, trovato={res.radius:.2f}, "
          f"centro atteso=(200,200), trovato=({res.center_x:.1f},{res.center_y:.1f}), "
          f"iterazioni={res.n_iterations}")
    assert abs(res.radius - 50.0) < 2.0
    assert abs(res.center_x - 200) < 1.0 and abs(res.center_y - 200) < 1.0

    # --- test measure_f_number_series con una serie di dischi a raggio crescente ---
    pix_size = 0.0074  # mm
    dist_step = 25.0  # mm
    true_slope = 1.2  # pix / mm  -> f_number atteso = 1/(2*slope*pix_size)
    images = []
    for i in range(5):
        r = 20 + true_slope * i * dist_step
        c = SimulatedCamera(nx=1200, ny=1200, radius_pix=r, peak_counts=1000, noise_std=2)
        images.append(c.acquire())
    fres = measure_f_number_series(images, pix_size_mm=pix_size, dist_step_mm=dist_step,
                                    sogl=0.3, s2_p2v=0.2)
    f_expected = 1.0 / (2 * true_slope * pix_size)
    print(f"  measure_f_number_series: F/# atteso={f_expected:.3f}, trovato={fres.f_number:.3f} "
          f"+/- {fres.f_number_error:.3f}")
    assert abs(fres.f_number - f_expected) / f_expected < 0.05

    # --- test make_spot / compute_frd_curve ---
    m = make_spot(100, 100, 50, 50, 30)
    assert abs(m.sum() - np.pi * 15**2) / (np.pi * 15**2) < 0.05

    curve = compute_frd_curve(frame, pix_size_mm=pix_size, dist_fiber_to_ccd_mm=17.5,
                              mode="ccd", fn_max=100.0, n_points=10)
    # il rumore di fondo puo' causare piccole oscillazioni punto a punto, ma il
    # trend complessivo deve essere crescente con l'apertura
    assert curve.intensity[-1] > curve.intensity[0]
    print("  compute_frd_curve: OK (intensità crescente con l'apertura)")

    # --- test funzioni di plotting (solo costruzione figure, no display) ---
    if go is not None:
        _ = plot_pupil_image(frame, res)
        _ = plot_f_number_fit(fres)
        _ = plot_centroid_migration(fres)
        _ = plot_frd_curve(curve)
        meas = [FrdMeasurement(label="test", image=frame)]
        batch = batch_frd_analysis(meas, pix_size_mm=pix_size, mode="ccd", n_points=10)
        _ = plot_frd_batch(batch)
        print("  funzioni di plotting: OK (figure costruite senza errori)")
    else:
        print("  plotly non installato: salto il test delle funzioni di plotting")

    print("Tutti i self-test superati.")
