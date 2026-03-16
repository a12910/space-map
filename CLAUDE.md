# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Space-map is a Python framework for reconstructing atlas-level single-cell 3D tissue maps from serial tissue sections. It combines multi-scale feature matching with LDDMM (Large-Deformation Diffeomorphic Metric Mapping) for two-stage registration: coarse affine alignment followed by fine non-rigid deformation.

## Development Commands

```bash
# Install in development mode
pip install -e .
pip install -e ".[dev]"      # includes pytest, black, flake8

# Verify installation
python -c "import space_map; print(space_map.__version__)"

# Run tests
pytest

# Lint / format
black space_map/
flake8 space_map/

# Build and serve documentation locally
pip install -e ".[docs]"
mkdocs serve                  # http://127.0.0.1:8000
```

## Architecture

### Two-Stage Registration Pipeline

1. **Affine Registration** (`space_map.affine`, `space_map.affine_block`) — coarse global alignment using density fields. Handles rotation, scaling, translation.
2. **LDDMM Registration** (`space_map.registration`) — fine non-rigid deformation. GPU-accelerated via PyTorch. Preserves topology.

### Key Modules

- **`space_map.flow`** — Main workflow orchestration. `FlowImport` initializes projects, `AutoFlowMultiCenter4`/`5` runs the registration pipeline, `FlowExport` exports results.
- **`space_map.base`** — Core data classes. `Slice` represents an individual tissue section; `SliceImg` handles image data; `SliceData` manages coordinate data.
- **`space_map.matches`** — Feature matching: SIFT, SIFT+VGG, LoFTR (deep learning). `MatchInit`/`MatchInitMulti` for initialization.
- **`space_map.registration`** — LDDMM implementations: `LDDMM2D`, `LDDMMBase`, `SVFLDDMM`.
- **`space_map.affine`** — Affine transforms: `AutoGrad`, `BestRotate`, `FilterGlobal`, `FilterGraph`, `FilterLabels`.
- **`space_map.utils`** — Visualization (`show`, `fig`), computation helpers, image processing, grid operations, Imaris export.
- **`space_map.find`** — Feature detection and error analysis.

### Data Flow & Key Constants

Slice objects track coordinates through registration stages via keys:
- `Slice.rawKey` — original input coordinates
- `Slice.align1Key` — after affine registration
- `Slice.align2Key` — after LDDMM registration
- `Slice.finalKey` — final output

### Runtime Project Directory Structure

When a registration project is initialized via `FlowImport`, it creates:
```
project_base/
├── conf.json       # Configuration (XYRANGE, XYD, slice IDs)
├── raw/            # Raw cell coordinates
├── imgs/           # Image data
├── outputs/        # Transformed coordinates
└── config.json     # Per-slice configuration
```

## Import Convention

The package directory is `space_map/` but the public import name is `spacemap`:
```python
import spacemap
from spacemap import Slice
```

## Code Style

- PEP 8, NumPy-style docstrings
- Formatting: `black`; Linting: `flake8`

## CI/CD

- **Tests** (`.github/workflows/test.yml`): Python 3.8–3.11 on Ubuntu + macOS
- **Docs** (`.github/workflows/docs.yml`): MkDocs build deployed to GitHub Pages on master push
- **Publish** (`.github/workflows/publish.yml`): PyPI publication

## Key Dependencies

numpy, pandas, torch, kornia, opencv-python, scipy, scikit-learn, scikit-image, numba, tifffile, nibabel, pycpd, matplotlib, seaborn
