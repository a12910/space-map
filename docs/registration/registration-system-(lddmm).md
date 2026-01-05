# Registration System (LDDMM)

The LDDMM (Large Deformation Diffeomorphic Metric Mapping) registration system is a core component of Space-map, responsible for precise non-rigid alignment of tissue sections while preserving topological structure.

For grid transformations, see: [Grid Transformations](../grid/grid-transformations.md).

## Overview

Space-map employs a two-stage registration strategy:

1. **Affine Registration (Coarse)**: Global linear transformation for initial alignment
2. **LDDMM Registration (Fine)**: Smooth, invertible non-rigid transformation for precise local alignment

This approach captures complex local deformations while maintaining tissue topology.

## How It Works in Practice

### In Your Workflow

When you run:

```python
mgr = spacemap.flow.AutoFlowMultiCenter4(slices, Slice.rawKey)
mgr.alignMethod = "auto"

# Step 1: Affine registration
mgr.affine("DF", show=True)

# Step 2: LDDMM registration
mgr.ldm_pair(Slice.align1Key, Slice.align2Key, show=True)
```

**What happens:**

1. **Affine stage** (`mgr.affine()`):
   - Converts cell coordinates to density field images
   - Finds feature correspondences (SIFT/LoFTR)
   - Computes affine transformation matrices
   - Applies transformations to align sections globally

2. **LDDMM stage** (`mgr.ldm_pair()`):
   - Takes affine-aligned images as input
   - Computes smooth deformation fields
   - Iteratively optimizes to match tissue structures
   - Preserves topology (no tears or folds)

## Registration Methods

### Affine Registration Methods

Control with `mgr.alignMethod`:

```python
# Automatic selection (recommended)
mgr.alignMethod = "auto"

# Traditional SIFT features
mgr.alignMethod = "sift"

# SIFT with VGG features (more robust)
mgr.alignMethod = "sift_vgg"

# Deep learning-based (requires more memory)
mgr.alignMethod = "loftr"
```

**How to choose:**
- Start with `"auto"` - works for most cases
- Try `"sift_vgg"` if results are poor
- Use `"loftr"` for challenging alignments (requires GPU)

### LDDMM Process

The `ldm_pair()` method handles LDDMM registration between adjacent sections:

```python
# Basic usage
mgr.ldm_pair(Slice.align1Key, Slice.align2Key, show=True)

# What it does:
# 1. Loads affine-aligned images (align1Key)
# 2. Computes velocity fields for smooth deformation
# 3. Iteratively optimizes matching
# 4. Saves results with align2Key
```

**GPU Acceleration:**
- Automatically uses GPU if available
- Check with: `print(f"Device: {mgr.gpu}")`
- Falls back to CPU if no GPU

## Multi-Scale Registration

LDDMM uses a multi-scale approach to avoid local optima:

### Registration Stages

The algorithm progresses through multiple stages with decreasing regularization:

| Stage | Purpose | v_scale | epsilon | iterations |
|-------|---------|---------|---------|------------|
| 1 | Coarse affine | 8.0 | 10000 | 300 |
| 2 | Medium affine | 4.0 | 1000 | 1000 |
| 3 | Fine affine | 1.0 | 50 | 6000 |
| 4 | Coarse LDDMM | 1.0 | 1000 | 20000 |
| 5 | Fine LDDMM | 1.0 | 1 | 20000 |

**Parameters:**
- **v_scale**: Controls deformation smoothness (higher = smoother)
- **epsilon**: Regularization strength (higher = more rigid)
- **iterations**: Optimization steps

## Data Flow Through Registration

```
Raw Coordinates
    ↓
Density Field Generation
    ↓
Affine Registration
    ↓ (Slice.align1Key)
Affine-Aligned Sections
    ↓
LDDMM Registration
    ↓ (Slice.align2Key)
Precisely Aligned Sections
    ↓
Export/Analysis
```

## Accessing Registration Results

### Get Transformed Points

```python
# After affine registration
points_affine = slice_obj.imgs[Slice.rawKey].get_points(Slice.align1Key)

# After LDDMM registration
points_lddmm = slice_obj.imgs[Slice.rawKey].get_points(Slice.align2Key)
```

### Compare Registration Quality

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original
ax = axes[0]
ax.scatter(points1_raw[:, 0], points1_raw[:, 1], s=1, alpha=0.3, c='red')
ax.scatter(points2_raw[:, 0], points2_raw[:, 1], s=1, alpha=0.3, c='blue')
ax.set_title('Before Registration')

# After affine
ax = axes[1]
ax.scatter(points1_affine[:, 0], points1_affine[:, 1], s=1, alpha=0.3, c='red')
ax.scatter(points2_affine[:, 0], points2_affine[:, 1], s=1, alpha=0.3, c='blue')
ax.set_title('After Affine')

# After LDDMM
ax = axes[2]
ax.scatter(points1_lddmm[:, 0], points1_lddmm[:, 1], s=1, alpha=0.3, c='red')
ax.scatter(points2_lddmm[:, 0], points2_lddmm[:, 1], s=1, alpha=0.3, c='blue')
ax.set_title('After LDDMM')

plt.show()
```

## Density Field Representation

Space-map converts sparse cell coordinates to continuous density fields for registration:

```
Cell Coordinates → Density Field → Registration
```

**"DF" Parameter:**
```python
mgr.affine("DF", show=True)
# "DF" = Density Field representation
```

This creates a grayscale image where pixel intensity represents cell density, enabling image-based registration techniques.

## Advanced: Manager Classes

Different managers handle the registration workflow:

### AutoFlowMultiCenter4 (Recommended)

```python
mgr = spacemap.flow.AutoFlowMultiCenter4(slices, Slice.rawKey)
```

**Features:**
- Starts from middle section
- Processes outward in both directions
- Better global consistency
- Reduces error accumulation

### AutoFlowMultiCenter5 (Latest)

```python
mgr = spacemap.flow.AutoFlowMultiCenter5(slices, Slice.rawKey)
```

**Features:**
- Additional optimizations
- Improved memory management
- Enhanced GPU utilization

### AutoFlowMultiCenter3 (Simple)

```python
mgr = spacemap.flow.AutoFlowMultiCenter3(slices)
```

**Features:**
- Sequential processing (0→1→2→...→N)
- Simpler implementation
- May accumulate errors

## Transformation Storage

Transformations are automatically saved in the project directory:

```
{BASE}/
└── {slice_id}/
    ├── H/              # Affine transformation matrices
    │   └── *.npz
    └── grids/          # LDDMM deformation grids
        └── *.npz
```

**Loading existing transformations:**
```python
# Projects automatically load saved transformations
flowImport = spacemap.flow.FlowImport(BASE)
slices = flowImport.slices  # Transformations restored
```

## Visualization

### Enable Visualization

```python
# Show alignment progress
mgr.affine("DF", show=True)
mgr.ldm_pair(Slice.align1Key, Slice.align2Key, show=True)
```

**What you'll see:**
- Side-by-side section comparisons
- Feature matches
- Transformation grids
- Alignment quality metrics

### Disable for Speed

```python
# Faster processing without visualization
mgr.affine("DF", show=False)
mgr.ldm_pair(Slice.align1Key, Slice.align2Key, show=False)
```

## Performance Considerations

### GPU vs CPU

```python
# Check device being used
print(f"Using device: {mgr.gpu}")

# GPU typically 10-50x faster for LDDMM
# CPU still works but slower
```

### Memory Usage

LDDMM memory usage depends on:
- Image resolution (controlled by `spacemap.XYD`)
- Number of iterations
- Batch size

**Reduce memory:**
```python
# Increase XYD (lowers resolution)
spacemap.XYD = 20  # Default is often 10-15
```

### Processing Time

Typical processing times (GPU):
- Affine: 1-5 seconds per pair
- LDDMM: 10-60 seconds per pair

Depends on image size, complexity, and hardware.

## Troubleshooting

### Poor Alignment

**Try different methods:**
```python
mgr.alignMethod = "sift_vgg"  # More robust features
```

**Adjust parameters:**
```python
spacemap.IMGCONF = {"raw": 0}  # Binary density field
```

### Memory Errors

```python
# Reduce resolution
spacemap.XYD = 25

# Process fewer sections
slices_subset = slices[:5]
```

### Slow Performance

```python
# Disable visualization
show=False

# Check GPU usage
print(f"GPU: {mgr.gpu}")

# Reduce iterations (less accurate)
# Not directly accessible - use default settings
```

## Integration with Spatial Analysis

After registration, you can analyze tissue in 3D:

```python
# Get aligned coordinates
for i, slice_obj in enumerate(slices):
    points_3d = slice_obj.imgs[Slice.rawKey].get_points(Slice.align2Key)
    z = i * z_spacing

    # Now analyze spatially with cell types, gene expression, etc.
```

## Complete Examples

See registration in action:
- [01_quickstart.ipynb](../../examples/01_quickstart.ipynb) - Basic registration
- [02_advanced_registration.ipynb](../../examples/02_advanced_registration.ipynb) - Quality assessment
- [Examples page](../examples/examples.md) - More scenarios

## Related Documentation

- [Key Workflows](../workflow/key-workflows.md) - Overall workflow
- [Data Management](../data/data-management.md) - Data structures
- [Grid Transformations](../grid/grid-transformations.md) - Transformation details 