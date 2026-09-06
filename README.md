# 🌙 Lunar AI

AI-based analysis of Chandrayaan-2 lunar surface imagery.

## Overview
This project processes and analyzes data from the following Chandrayaan-2 orbiter instruments:

- **IIR** — Imaging Infra-Red Spectrometer
- **OHR** — OHRC (Orbiter High Resolution Camera)
- **TMC** — Terrain Mapping Camera
- **LROC** — Quickmap reference data

## Repository Structure

```
lunar/
├── data/
│   ├── iirc/             # IIR (Imaging Infra-Red) datasets
│   ├── ohrc/             # OHRC (High Resolution Camera) datasets
│   ├── quickmap-lroc/    # LROC QuickMap reference
│   └── tmc/              # TMC (Terrain Mapping) datasets
├── notebook/             # Jupyter notebooks
│   └── v.ipynb
└── README.md
```

## Data
Raw imagery datasets are **not** tracked in Git (see `.gitignore`). Add them under `data/` locally or configure Git LFS if needed.

## Requirements
- Python 3.10+
- See `requirements.txt` (to be added)

## License
Add license information here.
