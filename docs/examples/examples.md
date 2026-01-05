# Examples

This page provides practical examples of using Space-map for different applications.

## Interactive Notebooks

For detailed, executable examples, see our Jupyter notebooks in the `examples/` directory:

- **[01_quickstart.ipynb](../../examples/01_quickstart.ipynb)**: Complete beginner-friendly tutorial
- **[02_advanced_registration.ipynb](../../examples/02_advanced_registration.ipynb)**: Advanced techniques and parameter tuning
- **[raw.ipynb](../../examples/raw.ipynb)**: Real-world data processing example

## Human Colon Polyp Study

Space-map was used to reconstruct 3D models of human colon tissues using spatial transcriptomics and proteomics data. The dataset included ~2.9M cells from Xenium and ~2.4M cells from CODEX.

```python
import spacemap
from spacemap import Slice
import pandas as pd

# Load Xenium data
xenium_df = pd.read_csv('xenium_data.csv')
# Expected columns: x, y, layer

# Organize by layers
xys = []
for layer_id in sorted(xenium_df['layer'].unique()):
    layer_data = xenium_df[xenium_df['layer'] == layer_id]
    xys.append(layer_data[['x', 'y']].values)

# Initialize project
BASE = "data/xenium_project"
flowImport = spacemap.flow.FlowImport(BASE)
flowImport.init_xys(xys)
slices = flowImport.slices

# Perform registration
mgr = spacemap.flow.AutoFlowMultiCenter4(slices, Slice.rawKey)
mgr.alignMethod = "auto"
mgr.affine("DF", show=True)
mgr.ldm_pair(Slice.align1Key, Slice.align2Key, show=True)

# Export results
export = spacemap.flow.FlowExport(slices)
# Export can be used for various output formats
```

## Example 1: Complete Registration Workflow with CSV Data

This example demonstrates a complete workflow for registering multiple tissue sections using cell coordinates from a CSV file.

```python
import spacemap
from spacemap import Slice
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load data from CSV file
# The CSV file contains cell coordinates with layer information
df = pd.read_csv("path/to/cells.csv.gz")

# Organize data by layers
xys = []
layer_ids = []

for layer_id in sorted(df['layer'].unique()):
    layer_data = df[df['layer'] == layer_id]
    xy = layer_data[['x', 'y']].values
    xys.append(xy)
    layer_ids.append(layer_id)

# Step 2: Initialize Space-map project
BASE = "data/flow"
flowImport = spacemap.flow.FlowImport(BASE)
flowImport.init_xys(xys, ids=layer_ids)
slices = flowImport.slices

print(f"Initialized {len(slices)} slices")
print(f"XYRANGE: {spacemap.XYRANGE}, XYD: {spacemap.XYD}")

# Step 3: Perform affine registration (coarse alignment)
mgr = spacemap.flow.AutoFlowMultiCenter4(slices, Slice.rawKey)
mgr.alignMethod = "auto"
mgr.affine("DF", show=True)

# Step 4: Perform LDDMM (fine alignment)
mgr.ldm_pair(Slice.align1Key, Slice.align2Key, show=True)

# Step 5: Visualize aligned sections in 3D
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

colors = plt.cm.viridis(np.linspace(0, 1, len(slices)))
z_spacing = 10

for i, slice_obj in enumerate(slices):
    # Get aligned points
    points = slice_obj.imgs[Slice.rawKey].get_points(Slice.align2Key)
    z = np.ones(len(points)) * i * z_spacing

    ax.scatter(points[:, 0], points[:, 1], z,
               color=colors[i], s=1, alpha=0.6, label=f"Layer {i}")

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z (Layer)')
ax.legend()
plt.title('3D Reconstruction of Aligned Tissue Sections')
plt.show()

# Step 6: Export results
export = spacemap.flow.FlowExport(slices)
# You can now use export object to save results in various formats
```

## Example 2: Working with CODEX Data

This example shows how to load and process CODEX spatial proteomics data.

```python
import spacemap
from spacemap import Slice
import pandas as pd

# Load CODEX data
# CODEX data typically has an 'array' column for layer identification
codex_df = pd.read_csv('codex_data.csv')
# Expected columns: x, y, array (layer identifier)

# Initialize project
BASE = "data/codex_project"
flowImport = spacemap.flow.FlowImport(BASE)

# Use the convenient CODEX initialization method
flowImport.init_from_codex('codex_data.csv')
slices = flowImport.slices

# Perform registration
mgr = spacemap.flow.AutoFlowMultiCenter4(slices, Slice.rawKey)
mgr.alignMethod = "auto"
mgr.affine("DF", show=True)
mgr.ldm_pair(Slice.align1Key, Slice.align2Key, show=True)

# Export results
export = spacemap.flow.FlowExport(slices)
print(f"Processed {len(slices)} CODEX sections")
```

## Example 3: Different Manager Classes

Space-map provides different manager classes for various use cases:

```python
import spacemap
from spacemap import Slice

# Initialize your data
flowImport = spacemap.flow.FlowImport(BASE)
flowImport.init_xys(xys, ids=layer_ids)
slices = flowImport.slices

# Option 1: AutoFlowMultiCenter4 (Recommended for most cases)
# Processes from middle outward for better global consistency
mgr = spacemap.flow.AutoFlowMultiCenter4(slices, Slice.rawKey)

# Option 2: AutoFlowMultiCenter5
# Latest version with additional optimizations
mgr = spacemap.flow.AutoFlowMultiCenter5(slices, Slice.rawKey)

# Option 3: AutoFlowMultiCenter3
# Simpler version for straightforward sequential processing
mgr = spacemap.flow.AutoFlowMultiCenter3(slices)

# All managers support the same workflow:
mgr.alignMethod = "auto"
mgr.affine("DF", show=True)
mgr.ldm_pair(Slice.align1Key, Slice.align2Key, show=True)
```

## Example 4: Customizing Registration Parameters

Fine-tune registration parameters for your specific data:

```python
import spacemap
from spacemap import Slice

# Initialize project
flowImport = spacemap.flow.FlowImport(BASE)

# Customize the ratio parameter (affects canvas size)
flowImport.ratio = 1.5  # Default is 1.4

flowImport.init_xys(xys, ids=layer_ids)
slices = flowImport.slices

# Check computed parameters
print(f"XYRANGE: {spacemap.XYRANGE}")  # Canvas size
print(f"XYD: {spacemap.XYD}")  # Grid spacing

# You can manually adjust these if needed
spacemap.XYRANGE = 2000
spacemap.XYD = 10

# Configure image processing
spacemap.IMGCONF = {"raw": 1}  # Use raw intensity values

# Create manager with custom settings
mgr = spacemap.flow.AutoFlowMultiCenter4(slices, Slice.rawKey)

# Choose alignment method
# Options: "auto", "sift", "sift_vgg", "loftr"
mgr.alignMethod = "sift_vgg"

# Perform registration
mgr.affine("DF", show=False)  # Set show=False for faster processing
mgr.ldm_pair(Slice.align1Key, Slice.align2Key, show=False)
```

## Example 5: Exporting Results

Different ways to export and save your aligned data:

```python
import spacemap
from spacemap import Slice
import pandas as pd
import numpy as np

# After registration is complete
export = spacemap.flow.FlowExport(slices)

# Method 1: Extract aligned points programmatically
aligned_data = []
z_spacing = 10

for i, slice_obj in enumerate(slices):
    # Get aligned coordinates
    points = slice_obj.imgs[Slice.rawKey].get_points(Slice.align2Key)

    # Add layer information
    layer_id = slice_obj.id
    z = i * z_spacing

    for point in points:
        aligned_data.append({
            'x': point[0],
            'y': point[1],
            'z': z,
            'layer': layer_id
        })

# Save to CSV
df_aligned = pd.DataFrame(aligned_data)
df_aligned.to_csv('aligned_cells.csv.gz', index=False, compression='gzip')

# Method 2: Save as numpy array
points_3d = np.array([[p[0], p[1], i*z_spacing]
                      for i, slice_obj in enumerate(slices)
                      for p in slice_obj.imgs[Slice.rawKey].get_points(Slice.align2Key)])
np.save('model_3d.npy', points_3d)

print(f"Exported {len(df_aligned)} cells")
```

## Tips and Best Practices

### For Large Datasets

- Use `show=False` in registration methods to speed up processing
- Consider processing subsets first to test parameters
- GPU acceleration is automatically used if available

### For Better Alignment

- Start with `alignMethod="auto"` and try other methods if results are not satisfactory
- Visualize intermediate results to catch issues early
- For noisy data, consider preprocessing to remove outliers

### Memory Management

- Reduce `spacemap.XYD` for lower memory usage (trade-off with precision)
- Process layers in batches for very large datasets
- Clean up intermediate results if memory is limited

## Additional Resources

For more detailed examples with full context and visualizations:

- **[01_quickstart.ipynb](../../examples/01_quickstart.ipynb)**: Complete beginner tutorial with visualizations
- **[02_advanced_registration.ipynb](../../examples/02_advanced_registration.ipynb)**: Advanced techniques and quality assessment
- **[raw.ipynb](../../examples/raw.ipynb)**: Real-world data processing

For detailed API documentation, see:

- [Data Management Guide](../data/data-management.md)
- [Registration System Documentation](../registration/registration-system-(lddmm).md)
- [API Reference](../api/api.md) 