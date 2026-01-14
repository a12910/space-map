# Space-map

**Reconstructing atlas-level single-cell 3D tissue maps from serial sections**

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://a12910.github.io/space-map)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/a12910/space-map/blob/master/License)
[![GitHub](https://img.shields.io/badge/GitHub-space--map-blue.svg)](https://github.com/a12910/space-map)

---

## Overview

Space-map is an open-source Python framework for reconstructing atlas-level single-cell 3D tissue maps from serial sections. It integrates single-cell coordinates with optional histological image features to assemble consecutive tissue sections into coherent 3D models.

High‑resolution three‑dimensional (3D) tissue atlases are transforming how we study cellular architecture‑function relationships in human tissues. Space-map addresses the challenge of accurate and efficient atlas-scale reconstruction by combining multi‑scale feature matching with large‑deformation diffeomorphic metric mapping (LDDMM), delivering global reconstructions while preserving local micro‑anatomy.

## Key Features

### Core Capabilities

- **Multi-modal Registration**: Combines cell coordinates, cell types, gene expression, and histological images for robust alignment
- **Two-stage Registration**: Efficient coarse alignment (affine) followed by precise fine registration (LDDMM)
- **Advanced Feature Matching**: Integrates deep learning (LoFTR) with traditional computer vision (SIFT)
- **GPU Acceleration**: Optimized LDDMM implementation for handling large-scale cellular data
- **Global Consistency**: Ensures structural coherence across non-adjacent sections
- **High Performance**: ~2-fold more accurate than existing methods (PASTE, STalign)
- **Scalable**: Designed for atlas-scale datasets with millions of cells

### Performance

- Handles datasets with 2-3 million cells
- Runs on standard laptop hardware
- GPU-accelerated for faster processing
- Memory-efficient processing pipeline

## Applications

Space-map has been successfully validated on:

- **Spatial Transcriptomics** (Xenium): ~2.9M cells across serial sections
- **Spatial Proteomics** (CODEX): ~2.4M cells across serial sections
- **Disease Models**: Colon polyp and reference colon 3D reconstructions

## Quick Example

```python
import space_map
from space_map import Slice
import pandas as pd

# Load spatial data
df = pd.read_csv("cells.csv.gz")

# Organize by layers
xys = []
layer_ids = []
for layer_id in sorted(df['layer'].unique()):
    layer_data = df[df['layer'] == layer_id]
    xy = layer_data[['x', 'y']].values
    xys.append(xy)
    layer_ids.append(layer_id)

# Initialize project
BASE = "data/flow"
flowImport = space_map.flow.FlowImport(BASE)
flowImport.init_xys(xys, ids=layer_ids)
slices = flowImport.slices

# Perform two-stage registration
mgr = space_map.flow.AutoFlowMultiCenter4(slices, Slice.rawKey)
mgr.alignMethod = "auto"
mgr.affine("DF", show=True)  # Coarse alignment
mgr.ldm_pair(Slice.align1Key, Slice.align2Key, show=True)  # Fine alignment

# Export results
export = space_map.flow.FlowExport(slices)
```

## Installation

### From Source (Recommended)

```bash
git clone https://github.com/a12910/space-map.git
cd space-map
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirement.txt
pip install -e .
```

See the [Installation Guide](getting-started/installation.md) for detailed instructions.

## Getting Started

1. **[Installation Guide](getting-started/installation.md)** - Set up Space-map on your system
2. **[Quick Start](getting-started/quickstart.md)** - Complete walkthrough with examples
3. **[Example Notebooks](https://github.com/a12910/space-map/tree/master/examples)** - Interactive tutorials

## Documentation

- **Getting Started**: Installation and quick start guides
- **Examples**: Jupyter notebooks with complete workflows
- **Contributing**: Guidelines for contributing to the project

## Methodology

Space-map employs a two-stage registration approach:

1. **Coarse Alignment (Affine)**: Fast global positioning using density field matching
   - Handles rotation, scaling, and translation
   - Computationally efficient
   - Avoids local optima

2. **Fine Alignment (LDDMM)**: Precise local deformations
   - Preserves micro-anatomical structures
   - GPU-accelerated
   - Maintains topological consistency

## System Requirements

- **Python**: 3.7 or higher
- **RAM**: 8GB minimum, 16GB+ recommended
- **GPU**: Optional but recommended for LDDMM acceleration
- **Storage**: Varies with dataset size

## Key Dependencies

- PyTorch - Deep learning and GPU acceleration
- OpenCV - Image processing and feature matching
- Kornia - Differentiable computer vision
- NumPy, Pandas, SciPy - Scientific computing
- Matplotlib, Seaborn - Visualization

## Authors & Affiliations

**Lead Authors** (contributed equally):
- **Rongduo Han** - Nankai University / Santa Clara University
- **Chenchen Zhu** - Stanford School of Medicine
- **Cihan Ruan** - Santa Clara University

**Principal Investigator**:
- **Michael Snyder** - Stanford School of Medicine (mpsnyder@stanford.edu)

**Contributing Authors**:
- Bingqing Zhao, Yuqi Tan, Emma Monte, Bei Wei, Joanna Bi, Thomas V. Karathanos, Rozelle Laquindanum, Greg Charville, Meng Wang, Yiing Lin, James M. Ford, Garry Nolan, Nam Ling

### Affiliations

- College of Software, Nankai University, Tianjin, China
- Department of Computer Science and Engineering, Santa Clara University, CA, USA
- Department of Genetics, Stanford School of Medicine, Stanford, CA, USA
- Department of Pathology, Stanford School of Medicine, Stanford, CA, USA

## Funding

This work was supported by:
- NIH Common Fund HuBMAP Program (U54HG010426, U54HG012723)
- NCI HTAN Program (U2CCA233311)
- HuBMAP JumpStart Fellowship (3OT2OD033759-01S3)
- AWS Cloud Credit for Research

## Citation

If you use Space-map in your research, please cite:

```
Han, R., Zhu, C., Ruan, C., et al. (2024). Space-map: Reconstructing
atlas-level single-cell 3D tissue maps from serial sections.
[Manuscript in preparation]
```

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/a12910/space-map/blob/master/License) file for details.

## Support & Contact

- **Issues**: [GitHub Issues](https://github.com/a12910/space-map/issues)
- **Discussions**: [GitHub Discussions](https://github.com/a12910/space-map/discussions)
- **Email**: a12910@qq.com

## Acknowledgments

We gratefully thank:
- Pauline Chu (Stanford Human Pathology/Histology Service Center) for tissue sectioning
- The HuBMAP Consortium for data and support
- AWS for computational resources
- All contributors and beta testers

---

**Ready to get started?** Check out the [Quick Start Guide](getting-started/quickstart.md) or explore our [example notebooks](https://github.com/a12910/space-map/tree/master/examples).
