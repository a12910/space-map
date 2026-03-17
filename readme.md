# Space-map

Space-map is an open-source framework for reconstructing atlas-level single-cell 3D tissue maps from serial sections. It integrates single-cell coordinates with optional histological image features to assemble serial sections into 3D models, combining multi-scale feature matching with large-deformation diffeomorphic metric mapping (LDDMM) to deliver global reconstructions while preserving local micro-anatomy.

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://a12910.github.io/space-map)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

- **Multi-modal Registration**: Combines cell coordinates, cell types, gene expression, and histological images for robust alignment
- **Two-stage Registration Approach**: Efficient coarse alignment followed by precise fine registration
- **Advanced Feature Matching**: Combines deep learning (LoFTR) with traditional computer vision methods (SIFT)
- **GPU-accelerated LDDMM**: Optimized for handling large-scale cellular data from multiple tissue sections
- **Global Consistency**: Ensures structural coherence between non-adjacent sections
- **High Performance**: ~2-fold more accurate than PASTE and STalign while running on standard laptop hardware

## Quick Start

### Installation

#### Option 1: Install from Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/a12910/space-map.git
cd space-map

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirement.txt

# Install Space-map in development mode
pip install -e .
```

#### Option 2: Direct Installation from GitHub

```bash
pip install git+https://github.com/a12910/space-map.git
```

> **Note**: Space-map will be available on PyPI soon. For now, please install from source.

### Basic Usage

```python
import spacemap
from spacemap import Slice
import pandas as pd

# Step 1: Load cell coordinate data
df = pd.read_csv("cells.csv.gz")

# Step 2: Organize data by layers
xys = []
layer_ids = []

for layer_id in sorted(df['layer'].unique()):
    layer_data = df[df['layer'] == layer_id]
    xy = layer_data[['x', 'y']].values
    xys.append(xy)
    layer_ids.append(layer_id)

# Step 3: Initialize project
BASE = "data/flow"
flowImport = spacemap.flow.FlowImport(BASE)
flowImport.init_xys(xys, ids=layer_ids)
slices = flowImport.slices

# Step 4: Perform registration
mgr = spacemap.flow.AutoFlowMultiCenter4(slices, Slice.rawKey)
mgr.alignMethod = "auto"  # Options: "auto", "sift", "sift_vgg", "loftr"
mgr.affine("DF", show=True)
mgr.ldm_pair(Slice.align1Key, Slice.align2Key, show=True)

# Step 5: Export results
export = spacemap.flow.FlowExport(slices)
```

For CODEX data:
```python
flowImport = spacemap.flow.FlowImport(BASE)
flowImport.init_from_codex('codex_data.csv')
slices = flowImport.slices
```

## Reproduce Results

A toy dataset is included so you can verify the pipeline in ~1 minute:

```bash
git clone https://github.com/a12910/space-map.git
cd space-map
pip install -e .
python benchmarks/run.py examples/toy_data.csv.gz
```

Or open the step-by-step notebook: [`benchmarks/example_notebook.ipynb`](benchmarks/example_notebook.ipynb)

For the full dataset (32 layers, ~2.9M cells, ~1 hour): `python benchmarks/run.py examples/cells2.csv.gz`

**Sample dataset with cell types** (20 layers, ~1.87M cells): download from [Releases](https://github.com/a12910/space-map/releases/download/v0.1.0/celltype_0427.csv.gz)

See [`benchmarks/README.md`](benchmarks/README.md) for details.

## Documentation

- **[Quick Start Guide](https://a12910.github.io/space-map/overview/quickstart/)** - Complete tutorial
- **[Examples](https://a12910.github.io/space-map/examples/examples/)** - Practical examples
- **[API Reference](https://a12910.github.io/space-map/api/api/)** - Detailed documentation

### Example Notebooks

- [`benchmarks/example_notebook.ipynb`](benchmarks/example_notebook.ipynb) - Reproduce alignment results
- [`examples/01_quickstart.ipynb`](examples/01_quickstart.ipynb) - Complete beginner tutorial
- [`examples/02_advanced_registration.ipynb`](examples/02_advanced_registration.ipynb) - Advanced techniques

## Applications

Space-map has been successfully applied to build high-resolution 3D tissue maps of:
- Serial sectioned spatial transcriptomics (Xenium, ~2.9M cells)
- Spatial proteomics dataset (CODEX, ~2.4M cells)
- 3D models for diseased (colon polyp) and reference colon

## Architecture

### Core Components

- **`flow.FlowImport`**: Data import and initialization
- **`flow.AutoFlowMultiCenter4/5`**: Registration workflow orchestration
- **`base.Slice`**: Individual tissue section management
- **`registration`**: LDDMM and deformable registration
- **`matches`**: Feature matching and alignment
- **`flow.FlowExport`**: Results export and visualization

### Project Structure

```
spacemap/
├── affine/              # Affine transformation code
├── affine_block/        # Block-based affine processing
├── base/                # Core classes (Slice, SliceImg, etc.)
├── flow/                # Flow-based registration pipeline
├── matches/             # Feature matching algorithms
├── registration/        # LDDMM registration
├── find/                # Feature detection and error analysis
└── utils/               # Utility functions
```

## Key Concepts

### Two-Stage Registration

1. **Affine Registration (Coarse)**: Global alignment using density fields
   - Fast and robust
   - Handles rotation, scaling, translation
   - Avoids local optima

2. **LDDMM (Fine)**: Local non-rigid deformation
   - Preserves topology
   - GPU-accelerated
   - Maintains micro-anatomical structures

### Data Keys

- `Slice.rawKey`: Original data
- `Slice.align1Key`: After affine registration
- `Slice.align2Key`: After LDDMM registration
- `Slice.finalKey`: Final output

## Requirements

Main dependencies:
- Python >= 3.7
- OpenCV
- NumPy
- PyTorch
- Kornia
- scikit-learn
- pandas
- matplotlib
- tifffile
- numba
- scipy
- tqdm

See [`requirement.txt`](requirement.txt) for complete list.

## Citation

If you use Space-map in your research, please cite our paper:

```bibtex
@article{spacemap2024,
  title={Space-map: Reconstructing atlas-level single-cell 3D tissue maps from serial sections},
  author={Han, Rongduo and Zhu, Chenchen and Ruan, Cihan and Snyder, Michael},
  journal={Nature Methods (under review)},
  year={2024}
}
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/overview/contributing.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](License) file for details.

## Support

- **Documentation**: https://a12910.github.io/space-map
- **Issues**: https://github.com/a12910/space-map/issues
- **Discussions**: https://github.com/a12910/space-map/discussions

## Authors

- **Rongduo Han** - Nankai University
- **Chenchen Zhu** - Stanford School of Medicine
- **Cihan Ruan** - Santa Clara University
- **Michael Snyder** - Stanford School of Medicine

See [full author list](https://a12910.github.io/space-map/#authors) for complete credits.

## Acknowledgments

This work was funded by:
- NIH Common Fund HuBMAP program (U54HG010426, U54HG012723)
- NCI HTAN program (U2CCA233311)
- HuBMAP JumpStart Fellowship (3OT2OD033759-01S3)
- AWS Cloud Credit for Research
