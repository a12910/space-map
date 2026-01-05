# Key Workflows

This page outlines the main processing workflows for 3D tissue reconstruction in Space-map, covering the entire process from raw data import to 3D model generation and spatial analysis.

For detailed examples, see [Examples](../examples/examples.md) and the interactive notebooks in `examples/`.

## Main Workflow Overview

Space-map employs a two-stage workflow, combining multi-scale feature matching with LDDMM (Large Deformation Diffeomorphic Metric Mapping) to balance efficiency and accuracy.

```
Data Import → Affine Registration → LDDMM Registration → Export Results
```

## Complete Workflow

### 1. Data Import and Initialization

Use `FlowImport` to load and initialize your data:

```python
import spacemap
from spacemap import Slice

# Initialize project
BASE = "data/flow"
flowImport = spacemap.flow.FlowImport(BASE)

# For standard CSV data with layer column
flowImport.init_xys(xys, ids=layer_ids)

# For CODEX data
flowImport.init_from_codex('codex_data.csv')

# Get slice objects
slices = flowImport.slices
```

**What happens:**
- Automatically calculates coordinate range (`XYRANGE`)
- Sets grid spacing (`XYD`)
- Creates `Slice` objects for each section
- Saves configuration to `conf.json`

### 2. Affine Registration (Coarse Alignment)

Use one of the `AutoFlowMultiCenter` managers:

```python
# Create registration manager
mgr = spacemap.flow.AutoFlowMultiCenter4(slices, Slice.rawKey)

# Set alignment method
mgr.alignMethod = "auto"  # Options: "auto", "sift", "sift_vgg", "loftr"

# Perform affine registration
mgr.affine("DF", show=True)
```

**Available Managers:**
- **`AutoFlowMultiCenter4`** (Recommended): Processes from middle outward for better global consistency
- **`AutoFlowMultiCenter5`**: Latest version with additional optimizations
- **`AutoFlowMultiCenter3`**: Simpler sequential processing

**Alignment Methods:**
- `"auto"`: Automatically selects best method (recommended)
- `"sift"`: Traditional SIFT features
- `"sift_vgg"`: SIFT with VGG features
- `"loftr"`: Deep learning-based LoFTR (requires more memory)

**What happens:**
- Generates density field representation from cell coordinates
- Finds feature correspondences between adjacent sections
- Computes affine transformations
- Results stored with key `Slice.align1Key`

### 3. LDDMM Registration (Fine Alignment)

Apply non-rigid deformation for precise local alignment:

```python
# Perform LDDMM between adjacent sections
mgr.ldm_pair(Slice.align1Key, Slice.align2Key, show=True)
```

**What happens:**
- Uses GPU acceleration if available
- Computes smooth, topology-preserving deformations
- Preserves micro-anatomical structures
- Results stored with key `Slice.align2Key`

### 4. Export and Visualization

Export aligned data and visualizations:

```python
# Create export manager
export = spacemap.flow.FlowExport(slices)

# Extract aligned points
for i, slice_obj in enumerate(slices):
    points = slice_obj.imgs[Slice.rawKey].get_points(Slice.align2Key)
    # Process points...
```

## Data Keys and Pipeline Stages

Space-map uses keys to track data through the registration pipeline:

| Key | Description | Stage |
|-----|-------------|-------|
| `Slice.rawKey` | Original input data | Initial |
| `Slice.align1Key` | After affine registration | Coarse alignment |
| `Slice.align2Key` | After LDDMM registration | Fine alignment |
| `Slice.finalKey` | Final output | Complete |

## Workflow Variations

### Sequential vs. Multi-center Registration

**Sequential (AutoFlowMultiCenter3):**
- Processes sections 0→1→2→...→N
- Simpler, but errors can accumulate

**Multi-center (AutoFlowMultiCenter4/5):**
- Starts from middle section
- Processes outward in both directions
- Better global consistency
- Recommended for most cases

### With vs. Without Visualization

```python
# With visualization (slower, useful for debugging)
mgr.affine("DF", show=True)
mgr.ldm_pair(Slice.align1Key, Slice.align2Key, show=True)

# Without visualization (faster, for production)
mgr.affine("DF", show=False)
mgr.ldm_pair(Slice.align1Key, Slice.align2Key, show=False)
```

## Advanced Workflows

### Custom Parameter Configuration

```python
# Customize canvas size
flowImport.ratio = 1.5  # Default: 1.4

# Manual parameter adjustment
spacemap.XYRANGE = 2000
spacemap.XYD = 10

# Image processing configuration
spacemap.IMGCONF = {"raw": 1}  # Use raw intensity values
```

### Processing Large Datasets

For datasets with many sections or cells:

1. **Reduce resolution:**
   ```python
   spacemap.XYD = 20  # Larger value = lower resolution
   ```

2. **Disable visualization:**
   ```python
   mgr.affine("DF", show=False)
   mgr.ldm_pair(Slice.align1Key, Slice.align2Key, show=False)
   ```

3. **Process in batches:**
   ```python
   # Process first 5 sections
   slices_batch = slices[:5]
   mgr = spacemap.flow.AutoFlowMultiCenter4(slices_batch, Slice.rawKey)
   ```

### Quality Assessment Workflow

```python
# Compare before and after alignment
for i in range(len(slices) - 1):
    # Before alignment
    points1_raw = slices[i].imgs[Slice.rawKey].get_points(Slice.rawKey)
    points2_raw = slices[i+1].imgs[Slice.rawKey].get_points(Slice.rawKey)

    # After alignment
    points1_aligned = slices[i].imgs[Slice.rawKey].get_points(Slice.align2Key)
    points2_aligned = slices[i+1].imgs[Slice.rawKey].get_points(Slice.align2Key)

    # Visualize or compute metrics...
```

## Integration with Spatial Analysis

### Cell Type Analysis

```python
# After registration, analyze cell types spatially
for i, slice_obj in enumerate(slices):
    points = slice_obj.imgs[Slice.rawKey].get_points(Slice.align2Key)
    # Match with original metadata
    # Analyze spatial patterns...
```

### Gene Expression Mapping

```python
# Map gene expression to 3D coordinates
for i, slice_obj in enumerate(slices):
    points = slice_obj.imgs[Slice.rawKey].get_points(Slice.align2Key)
    z = i * z_spacing
    # Add gene expression data
    # Create 3D visualization...
```

## Best Practices

1. **Always start with a small test dataset** to validate parameters
2. **Use `show=True` initially** to verify alignment quality
3. **Try different `alignMethod` options** if results are poor
4. **Check intermediate results** between affine and LDDMM stages
5. **Save your work frequently** - transformations are stored automatically
6. **Document your parameters** for reproducibility

## Troubleshooting Workflows

### Poor Alignment Results

1. Try different alignment methods:
   ```python
   mgr.alignMethod = "sift_vgg"  # or "loftr"
   ```

2. Adjust preprocessing:
   ```python
   spacemap.IMGCONF = {"raw": 0}  # Binary density field
   ```

3. Check data quality and remove outliers

### Memory Issues

1. Reduce resolution:
   ```python
   spacemap.XYD = 20  # or higher
   ```

2. Process fewer sections at once

3. Disable visualization:
   ```python
   show=False
   ```

### Slow Processing

1. Ensure GPU is being used:
   ```python
   print(f"Device: {mgr.gpu}")
   ```

2. Disable visualization

3. Reduce data resolution

## Complete Example

See the complete workflow in action:
- [01_quickstart.ipynb](../../examples/01_quickstart.ipynb) - Basic workflow
- [02_advanced_registration.ipynb](../../examples/02_advanced_registration.ipynb) - Advanced techniques

## Related Documentation

- [Data Management](../data/data-management.md) - Data structures and storage
- [Registration System](../registration/registration-system-(lddmm).md) - LDDMM details
- [Examples](../examples/examples.md) - Practical examples 