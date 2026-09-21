# frd_toolkit.py — Criticità da affrontare

Revisione del porting Python del pacchetto IDL FRD (A. Tozzi, INAF-Arcetri, 2008-2012).
Il self-test incluso nel file passa; le criticità sotto non sono coperte dal self-test.

Priorità: **Alta** = influenza direttamente i risultati scientifici; **Media** = rischio di uso scorretto o risultati fuorvianti; **Bassa** = pulizia/robustezza.

---

## 1. Coordinate errate in modalità `split` — Alta (bug verificato)

**Dove:** `analyze_pupil(split=True)`, righe ~291-311.

**Problema:** la docstring dichiara che i centri vengono riportati nel sistema dell'immagine originale, ma gli `offsets` `(0,d), (c,d), (0,0), (c,0)` sono relativi al riquadro ritagliato che parte da `(cx-c, cy-d)`. I centri risultano quindi sfasati di `(cx-c, cy-d)`.

**Verifica:** immagine 1000×800, quattro dischi r=40 px centrati in (450,510), (650,510), (450,310), (650,310), `sogl=0.3, s2_p2v=0.2`. Centri restituiti: (300,420), (500,420), (300,220), (500,220) → offset (150, 90) = (cx-c, cy-d). Raggi corretti (40.0).

**Impatto:** le distanze relative (e quindi `quadrant_distances`) sono corrette; le posizioni assolute no.

**Correzione proposta:** `offsets = [(cx-c, cy), (cx, cy), (cx-c, cy-d), (cx, cy-d)]`, più un test su dati sintetici con centri noti.

**Note collegate:**
- centro dello split hardcoded a `(nx/2 + 50, ny/2 + 10)`, specifico del vecchio banco → renderlo parametro obbligatorio o stimarlo dall'immagine;
- la docstring riporta due descrizioni incoerenti dell'ordine dei quadranti (`[dx-basso, sx-basso, ...]` vs `k[0] = -x+y`).

---

## 2. Normalizzazione della curva FRD all'apertura massima disponibile — Alta

**Dove:** `FrdCurve.normalized()`, `compute_frd_curve()`.

**Problema:** la curva è normalizzata al flusso dentro l'ultima apertura, il cui diametro in modalità `'pupil'` dipende dalla distanza del centro dal bordo del sensore (`max_diam = 2·min(cx, cy, nx-cx, ny-cy)`). Misure con centri diversi vengono quindi normalizzate a F/# diversi e non a un flusso totale comune.

**Impatto:** confronti tra misure (`plot_frd_batch`) potenzialmente falsati; l'energia inclusa non è riferita al flusso totale.

**Correzione proposta:** normalizzare a un F/# di riferimento fisso comune a tutte le misure (o al flusso totale stimato con fondo sottratto), esplicitandolo come parametro; usare una griglia di F/# comune a tutte le misure invece di una griglia di diametri dipendente dall'immagine.

---

## 3. Assenza di stima/sottrazione del fondo residuo — Alta

**Dove:** `compute_frd_curve()`, `batch_frd_analysis()`.

**Problema:** viene sottratto solo il dark (se fornito). Un piedistallo residuo (luce diffusa, offset, dark non rappresentativo) fa crescere il flusso integrato come D², con effetto massimo proprio sul punto usato per la normalizzazione (vedi punto 2).

**Correzione proposta:** stimare il fondo per pixel da un anello esterno (sigma-clipped median) e sottrarlo prima dell'integrazione; riportare il livello stimato nel risultato.

---

## 4. Dipendenza dalla distanza assoluta fibra-sensore — Media

**Dove:** `compute_frd_curve()` (`dist_fiber_to_ccd_mm`, default 17.5 mm).

**Problema:** F/# = L / (D · pix_size): ogni errore relativo su L si propaga 1:1 sul F/#. Il metodo differenziale di `measure_f_number_series` non ha questo problema.

**Correzione proposta:** documentare l'incertezza su L e propagarla; valutare la calibrazione di L tramite la serie a distanze note (stesso setup).

---

## 5. Raggio dipendente dalla soglia e default incoerenti — Media

**Dove:** `_analyze_single_pupil()`, `measure_f_number_series()`, `compute_frd_curve()`.

**Problema:** per profili di campo lontano reali (bordi morbidi, non top-hat) il raggio "trovato" dipende dalla soglia; nel metodo a serie l'effetto si cancella solo se il profilo scala in modo auto-simile con la distanza. I default sono incoerenti:

| Contesto | `sogl` | `s2_p2v` |
|---|---|---|
| `analyze_pupil` | 0.05 | None (media sopra s1) |
| `measure_f_number_series` | 0.3 | 0.02 |
| `compute_frd_curve` | 0.1 | 0.2 |
| self-test | 0.3 | 0.2 |

Inoltre, con `s2_p2v` esplicito, `s2 = (max - min)·frac` non aggiunge l'offset `min` (a differenza di `s1`).

**Correzione proposta:** definire il raggio in modo esplicito e fisicamente motivato (es. raggio a EE fissata, o FWHM/frazione del plateau), con default unici in configurazione; test su profili non top-hat.

---

## 6. Metrica FRD incompleta: `input_f_number` inutilizzato — Media

**Dove:** `FrdMeasurement`, `batch_frd_analysis()`.

**Problema:** il F/# di iniezione viene memorizzato ma mai usato. Manca il confronto F_in / F_out (es. F/# di uscita a EE fissata, 90-95%, in funzione di F_in), che è la metrica FRD standard.

**Correzione proposta:** funzione che estrae F_out a EE fissata dalla curva (interpolazione) e grafico F_out vs F_in.

---

## 7. Criticità minori — Bassa

- **Cubi FITS:** `load_image` restituisce un array 3D per FITS multi-frame, ma `compute_frd_curve` accetta solo 2D (`ny, nx = image.shape`). Aggiungere media/mediana sui frame o un controllo esplicito.
- **Discretizzazione:** diametri arrotondati all'intero e maschere campionate al centro pixel (`make_spot`), senza sub-pixel → gradini nella curva a piccoli diametri.
- **Griglia:** i diametri partono da `fn_diam_min + step`, quindi il punto a `fn_max` non è mai incluso.
- **`quadrant_distances`:** il commento sulle "diagonali" è errato (vengono calcolati 4 lati); la diagonale è solo √2·lato medio (ipotesi di quadrato non verificata).
- **`generate_synthetic_frd_images`:** i parametri `radius_*` sono in realtà sigma gaussiani; il `np.clip(…, 0)` distorce la statistica del rumore; la funzione non è usata nel self-test.
- **`acquire_with_dark`:** dark da un singolo frame (rumoroso rispetto a una media di N frame di scienza).
- **Tilt:** conversione in piccolo angolo (b·pix_size ≈ rad), accettabile ma da documentare insieme alla convenzione di segno.

---

## Riferimenti

- Ramsey L. W., 1988, *ASP Conf. Ser.* 3, 26 — metodo della curva di energia inclusa vs F/#.
- Carrasco E., Parry I. R., 1994, *MNRAS* 271, 1 — caratterizzazione FRD delle fibre.
