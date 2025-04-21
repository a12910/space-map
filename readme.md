# Space-map

SpaceMap is an open-source framework for reconstructing atlas-level single-cell 3D tissue maps from serial sections. It integrates single-cell coordinates with optional histological image features to assemble serial sections into 3D models, combining multi-scale feature matching with large-deformation diffeomorphic metric mapping (LDDMM) to deliver global reconstructions while preserving local micro-anatomy.

## Key Features

- **Multi-modal Registration**: Combines cell coordinates, cell types, gene expression, and histological images for robust alignment
- **Two-stage Registration Approach**: Efficient coarse alignment followed by precise fine registration
- **Advanced Feature Matching**: Combines deep learning (LoFTR) with traditional computer vision methods (SIFT)
- **GPU-accelerated LDDMM**: Optimized for handling large-scale cellular data from multiple tissue sections
- **Global Consistency**: Ensures structural coherence between non-adjacent sections
- **High Performance**: ~2-fold more accurate than PASTE and STalign while running on a standard laptop

## Installation

### Requirements

```bash
pip install -r requirement.txt
```

Main dependencies include:
- OpenCV
- NumPy
- PyTorch
- Kornia
- scikit-learn
- tifffile
- matplotlib
- numba
- scipy
- tqdm
- nibabel
- seaborn
- scikit-image

## Project Structure

- **affine/**: Affine transformation related code
- **affine_block/**: Block processing for affine transformations
- **base/**: Base classes and functions
- **flow/**: Flow-based transformation code
- **matches/**: Feature matching related code
- **registration/**: Image registration algorithms
- **find/**: Feature point detection and error analysis
- **utils/**: Utility functions

## Usage Example

```python
import spacemap as sm

# Load image data
img = sm.utils.img.load_image("path/to/image.tif")

# Visualize data
sm.utils.show.plot_image(img)

# Apply spatial transformation
transformed_data = sm.flow.transform(data, params)

# Detect keypoints
keypoints = sm.find.detect_keypoints(img)
```

## Applications

SpaceMap has been successfully applied to build high-resolution 3D tissue maps of:
- Serial sectioned spatial transcriptomics (Xenium, ~2M cells)
- Spatial proteomics dataset (CODEX, ~1M cells)
- 3D models for diseased (colon polyp) and reference colon
