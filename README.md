# frd_analysis

Focal Ratio Degradation analysis of FITS pupil images, built on `lib/frd_toolkit.py`.

```
frd_analysis/
├── frd_gui.py               # Tkinter GUI (entry point)
├── frd_config.yaml          # default parameters, loaded by the GUI at start-up
├── frd_toolkit_criticita.md # known issues of frd_toolkit.py, to be addressed
└── lib/
    ├── frd_pipeline.py      # analysis driver and output writer
    ├── frd_io.py            # FITS discovery, loading, distance assignment
    └── frd_toolkit.py       # computational library (port of the IDL package)
```

Run from the `frd_analysis/` directory:

- GUI: `python frd_gui.py`
- no GUI: `python -m lib.frd_pipeline frd_config.yaml`

Dependencies: numpy, scipy, astropy, pyyaml, plotly (plots), kaleido (optional, PNG plots), tkinter.

## Copyright

- Original FRD IDL package (2008-2012) and its Python port `lib/frd_toolkit.py`: Copyright (c) Andrea Tozzi (INAF - Osservatorio Astrofisico di Arcetri).
- Modifications and new modules (GUI, pipeline, I/O): Copyright (c) 2026 Alessio Turchi (INAF - Osservatorio Astrofisico di Arcetri).

## License

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. It is distributed WITHOUT ANY WARRANTY; see the `LICENSE` file for
the full text.
