# Quick Start Guide

This guide walks you through a complete 3D tissue reconstruction workflow using Space-map.

## Overview

Space-map uses a two-stage registration approach to align serial tissue sections:

1. **Affine Registration** - Fast coarse alignment (handles rotation, scaling, translation)
2. **LDDMM Registration** - Precise fine alignment (preserves micro-anatomy)

## Prerequisites

- Space-map installed ([Installation Guide](installation.md))
- Spatial data with cell coordinates and layer/section identifiers
- Basic Python knowledge

## Data Format

Your input data should contain:

- **x, y coordinates** - Cell positions within each section
- **layer/section ID** - Identifier for each tissue section

Example CSV format:

```csv
x,y,layer
100.5,200.3,0
150.2,180.7,0
95.3,205.1,1
...
```

## Complete Workflow

### Step 1: Import Libraries

```python
import space_map
from space_map import Slice
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Check installation
print(f"Space-map version: {space_map.__version__}")
```

### Step 2: Load and Organize Data

```python
# Load cell coordinates from file
df = pd.read_csv("path/to/cells.csv.gz")

# Organize data by tissue sections
xys = []  # List to store coordinate arrays for each layer
layer_ids = []  # List to store layer identifiers

for layer_id in sorted(df['layer'].unique()):
    # Extract coordinates for this layer
    layer_data = df[df['layer'] == layer_id]
    xy = layer_data[['x', 'y']].values  # Extract x, y as numpy array

    xys.append(xy)
    layer_ids.append(layer_id)

print(f"Loaded {len(xys)} tissue sections")
print(f"Total cells: {sum(len(xy) for xy in xys):,}")
```

### Step 3: Initialize Space-map Project

```python
# Set project directory (will be created if doesn't exist)
BASE = "data/my_reconstruction"

# Initialize the Flow system for data management
flowImport = space_map.flow.FlowImport(BASE)

# Initialize slices with your coordinate data
flowImport.init_xys(xys, ids=layer_ids)

# Get slice objects for processing
slices = flowImport.slices

print(f"Initialized {len(slices)} slices")
```

### Step 4: Affine Registration (Coarse Alignment)

```python
# Create registration manager
# AutoFlowMultiCenter4 processes from center outward for better results
mgr = space_map.flow.AutoFlowMultiCenter4(slices, Slice.rawKey)

# Set alignment method
# "auto" - Automatic selection (recommended)
# "sift" - Traditional SIFT features
# "sift_vgg" - SIFT with VGG feature enhancement
# "loftr" - Deep learning feature matching
mgr.alignMethod = "auto"

# Perform affine registration using Density Field (DF) representation
# show=True displays alignment visualizations (set False for faster processing)
print("Starting affine registration...")
mgr.affine("DF", show=True)

print("✓ Affine registration completed!")
```

**What happens in this step:**
- Converts cell coordinates to density fields
- Detects and matches features between adjacent sections
- Computes optimal affine transformations (rotation, scaling, translation)
- Applies transformations for coarse alignment

### Step 5: LDDMM Registration (Fine Alignment)

```python
# Perform LDDMM registration for precise local deformations
# This step preserves micro-anatomical structures
print("Starting LDDMM registration...")
mgr.ldm_pair(Slice.align1Key, Slice.align2Key, show=True)

print("✓ LDDMM registration completed!")
```

**What happens in this step:**
- Takes affine-aligned sections as input
- Computes smooth, diffeomorphic deformations
- Preserves topology and local structure
- GPU-accelerated for performance

### Step 6: Visualize 3D Reconstruction

```python
# Create 3D visualization
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Use different colors for each layer
colors = plt.cm.viridis(np.linspace(0, 1, len(slices)))
z_spacing = 10  # Distance between sections (adjust as needed)

# Plot each aligned section
for i, slice_obj in enumerate(slices):
    # Get final aligned coordinates
    points = slice_obj.imgs[Slice.rawKey].get_points(Slice.align2Key)

    # Downsample for visualization if needed
    if len(points) > 10000:
        indices = np.random.choice(len(points), 10000, replace=False)
        points = points[indices]

    # Create z-coordinates for this section
    z = np.ones(len(points)) * i * z_spacing

    # Plot points
    ax.scatter(points[:, 0], points[:, 1], z,
               c=[colors[i]], s=1, alpha=0.5,
               label=f"Section {slice_obj.id}")

# Customize plot
ax.set_xlabel('X (μm)')
ax.set_ylabel('Y (μm)')
ax.set_zlabel('Z (Section)')
ax.set_title('3D Tissue Reconstruction', fontsize=16, fontweight='bold')
ax.view_init(elev=20, azim=45)

# Add legend (limit to first few sections if many)
if len(slices) <= 10:
    ax.legend(markerscale=5)

plt.tight_layout()
plt.show()
```

### Step 7: Export Results

```python
# Initialize export manager
export = space_map.flow.FlowExport(slices)

# Export aligned coordinates to CSV
aligned_data = []

for i, slice_obj in enumerate(slices):
    # Get aligned points
    points = slice_obj.imgs[Slice.rawKey].get_points(Slice.align2Key)
    z = i * z_spacing

    # Create records
    for point in points:
        aligned_data.append({
            'x': point[0],
            'y': point[1],
            'z': z,
            'section_id': slice_obj.id
        })

# Save to compressed CSV
df_aligned = pd.DataFrame(aligned_data)
output_file = f'{BASE}/aligned_coordinates.csv.gz'
df_aligned.to_csv(output_file, index=False, compression='gzip')

print(f"✓ Exported {len(df_aligned):,} aligned cells to {output_file}")
```

## Working with Different Data Types

### CODEX Spatial Proteomics Data

For CODEX data with 'array' column indicating sections:

```python
flowImport = space_map.flow.FlowImport(BASE)
flowImport.init_from_codex('codex_data.csv')
slices = flowImport.slices
```

### Custom Data Loading

For other formats:

```python
# Manual data preparation
xys = []
for section_file in section_files:
    coords = load_your_data(section_file)  # Your custom loader
    xys.append(coords)

flowImport.init_xys(xys, ids=range(len(xys)))
```

## Advanced Options

### Alternative Registration Managers

```python
# Option 1: AutoFlowMultiCenter4 (Recommended)
# Processes from center section outward
mgr = space_map.flow.AutoFlowMultiCenter4(slices, Slice.rawKey)

# Option 2: AutoFlowMultiCenter5 (Latest)
# Includes additional optimizations
mgr = space_map.flow.AutoFlowMultiCenter5(slices, Slice.rawKey)
```

### Alignment Method Selection

```python
# Automatic (recommended) - selects best method
mgr.alignMethod = "auto"

# SIFT - Fast, works well for most data
mgr.alignMethod = "sift"

# SIFT+VGG - More robust for challenging alignments
mgr.alignMethod = "sift_vgg"

# LoFTR - Deep learning, best for difficult cases
mgr.alignMethod = "loftr"
```

### Performance Tuning

```python
# Disable visualization for faster processing
mgr.affine("DF", show=False)
mgr.ldm_pair(Slice.align1Key, Slice.align2Key, show=False)

# For large datasets, adjust grid spacing
import space_map
space_map.XYD = 20  # Larger value = coarser grid = less memory
```

## Understanding Data Keys

Space-map tracks transformations using keys:

- **`Slice.rawKey`** - Original input coordinates
- **`Slice.align1Key`** - After affine registration
- **`Slice.align2Key`** - After LDDMM registration (final)

Access coordinates at any stage:

```python
# Original coordinates
raw_coords = slice_obj.imgs[Slice.rawKey].get_points(Slice.rawKey)

# After affine alignment
affine_coords = slice_obj.imgs[Slice.rawKey].get_points(Slice.align1Key)

# Final aligned coordinates
final_coords = slice_obj.imgs[Slice.rawKey].get_points(Slice.align2Key)
```

## Tips for Success

1. **Start Small** - Test with 3-5 sections first before processing entire dataset
2. **Check Intermediate Results** - Use `show=True` to verify alignment quality
3. **GPU Acceleration** - Space-map automatically uses GPU if available
4. **Memory Management** - For very large datasets (>5M cells), increase `space_map.XYD`
5. **Quality Control** - Visually inspect alignments and try different methods if needed

## Common Issues

**Q: Registration is very slow**

A:
- Set `show=False` to disable visualizations
- Ensure PyTorch can use GPU: `torch.cuda.is_available()`
- Reduce resolution with larger `space_map.XYD` value

**Q: Alignment looks poor**

A:
- Try different alignment methods (`"sift_vgg"` or `"loftr"`)
- Check if sections are in correct order
- Verify coordinate units are consistent across sections

**Q: Out of memory errors**

A:
- Increase `space_map.XYD` (e.g., from 10 to 20)
- Process fewer sections at a time
- Close visualization windows to free memory

**Q: Import errors**

A:
- Verify installation: `python -c "import space_map"`
- Reinstall dependencies: `pip install -r requirement.txt`

## Example Notebooks

For interactive tutorials with full visualizations:

- [01_quickstart.ipynb](https://github.com/a12910/space-map/blob/master/examples/01_quickstart.ipynb) - Complete beginner tutorial
- [02_advanced_registration.ipynb](https://github.com/a12910/space-map/blob/master/examples/02_advanced_registration.ipynb) - Advanced techniques

## Next Steps

- Explore [example notebooks](https://github.com/a12910/space-map/tree/master/examples) with real data
- Read about [contributing](../contributing.md) to Space-map
- Check [GitHub Issues](https://github.com/a12910/space-map/issues) for support

## Getting Help

Need assistance?

- **GitHub Issues**: [Report bugs or ask questions](https://github.com/a12910/space-map/issues)
- **GitHub Discussions**: [Community Q&A](https://github.com/a12910/space-map/discussions)
- **Email**: a12910@qq.com

---

**Ready to process your own data?** Start with a small subset and gradually scale up. Good luck with your 3D reconstruction!
