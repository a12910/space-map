# Data Management

The data management system in Space-map provides infrastructure for importing, organizing, processing, and exporting tissue section data throughout the 3D reconstruction workflow.

For registration details, see: [Registration System (LDDMM)](../registration/registration-system-(lddmm).md).

## Overview

Space-map's data management revolves around three core components:

- **`FlowImport`**: Data import and project initialization
- **`Slice`**: Individual tissue section management
- **`FlowExport`**: Results export and visualization

Together, these components handle the complete data lifecycle from raw input to final 3D models.

## Core Classes

### FlowImport

Initializes the data processing workflow and sets up the project environment.

```python
import spacemap

BASE = "data/flow"
flowImport = spacemap.flow.FlowImport(BASE)
```

#### Key Methods

**`init_xys(xys, ids=None)`**

Initialize slices from coordinate arrays:

```python
# xys: list of numpy arrays (N×2) with x,y coordinates
# ids: optional list of layer identifiers

flowImport.init_xys(xys, ids=layer_ids)
slices = flowImport.slices
```

**What it does:**
- Calculates optimal coordinate range (`XYRANGE`)
- Determines grid spacing (`XYD`)
- Creates `Slice` objects for each section
- Saves configuration to `{BASE}/conf.json`
- Creates raw data backup in `{BASE}/raw/cells.csv.gz`

**`init_from_codex(csvPath)`**

Special method for CODEX spatial proteomics data:

```python
flowImport.init_from_codex('codex_data.csv')
slices = flowImport.slices
```

Automatically handles CODEX's 'array' column for layer identification.

**`auto_init()`**

Reload an existing project:

```python
# Automatically called on FlowImport initialization
# Reads conf.json and reconstructs slice objects
flowImport = spacemap.flow.FlowImport(BASE)  # Loads existing project
```

#### Configuration Parameters

After initialization, check the computed parameters:

```python
print(f"XYRANGE: {spacemap.XYRANGE}")  # Canvas size
print(f"XYD: {spacemap.XYD}")          # Grid spacing
```

You can customize these before initialization:

```python
flowImport.ratio = 1.5  # Adjust canvas size (default: 1.4)
```

### Slice

Manages individual tissue sections and their transformations.

```python
from spacemap import Slice

# Access slices from flowImport
for slice_obj in slices:
    print(f"Slice ID: {slice_obj.id}")
```

#### Data Keys

Slices track data through the registration pipeline using keys:

```python
# Predefined keys
Slice.rawKey       # "raw" - Original data
Slice.align1Key    # "align1" - After affine registration
Slice.align2Key    # "align2" - After LDDMM registration
Slice.finalKey     # "final" - Final output
Slice.enhanceKey   # "enhance" - Enhanced version
```

#### Accessing Data

**Get point coordinates:**

```python
# Get points at specific stage
points_raw = slice_obj.imgs[Slice.rawKey].get_points(Slice.rawKey)
points_aligned = slice_obj.imgs[Slice.rawKey].get_points(Slice.align2Key)
```

**Create images:**

```python
# Generate density field image
img = slice_obj.create_img(
    useKey=Slice.rawKey,     # Data source
    dfKey=Slice.align2Key,   # Transformation to apply
    scale=True,              # Scale to XYRANGE/XYD
    fixHe=True              # H&E normalization
)
```

### FlowExport

Export aligned data and results.

```python
export = spacemap.flow.FlowExport(slices)
```

#### Export Methods

**Extract aligned coordinates:**

```python
# Manual extraction
for i, slice_obj in enumerate(slices):
    points = slice_obj.imgs[Slice.rawKey].get_points(Slice.align2Key)
    # Save or process points...
```

**Export images (if available):**

```python
# Export all section images
imgs = export.export_imgs(
    "DF",               # Image type key
    Slice.align2Key,    # Transformation stage
    False               # Return as array
)
```

## Project Structure

When you initialize a project, Space-map creates this structure:

```
{BASE}/
├── conf.json           # Global configuration
│   ├── XYD             # Grid spacing
│   ├── XYRANGE         # Canvas size
│   └── ids             # Layer identifiers
├── raw/
│   └── cells.csv.gz    # Backup of raw data
├── outputs/
│   └── {slice_id}_{key}_{stage}.csv.gz  # Transformed coordinates
├── imgs/
│   └── {slice_id}_{key}_{stage}.png     # Generated images
└── {slice_id}/
    ├── H/              # Affine transformations
    └── grids/          # LDDMM grids
```

## Data Flow

Understanding how data flows through the system:

```
Raw CSV Data
    ↓
FlowImport.init_xys()
    ↓
Slice objects created with rawKey data
    ↓
AutoFlowMultiCenter4.affine()
    ↓
Slice objects now have align1Key data
    ↓
AutoFlowMultiCenter4.ldm_pair()
    ↓
Slice objects now have align2Key data
    ↓
FlowExport / Manual extraction
    ↓
3D Model / Analysis
```

## Working with Data

### Loading Existing Projects

```python
# Load an existing project
BASE = "data/flow"
flowImport = spacemap.flow.FlowImport(BASE)
slices = flowImport.slices  # Automatically restored

print(f"Loaded {len(slices)} slices")
print(f"XYRANGE: {spacemap.XYRANGE}")
print(f"XYD: {spacemap.XYD}")
```

### Accessing Transformation History

```python
# Check available data keys for a slice
slice_obj = slices[0]

# Try to get points at different stages
try:
    points_raw = slice_obj.imgs[Slice.rawKey].get_points(Slice.rawKey)
    print(f"Raw points: {len(points_raw)}")
except:
    print("Raw data not available")

try:
    points_aligned = slice_obj.imgs[Slice.rawKey].get_points(Slice.align2Key)
    print(f"Aligned points: {len(points_aligned)}")
except:
    print("Aligned data not available yet")
```

### Saving Custom Data

```python
# Save custom point transformations
custom_points = ...  # Your transformed points
slice_obj.imgs[Slice.rawKey].save_points(custom_points, "custom_key")

# Later retrieve them
loaded_points = slice_obj.imgs[Slice.rawKey].get_points("custom_key")
```

## Advanced Usage

### Custom Canvas Configuration

```python
# Initialize with custom parameters
flowImport = spacemap.flow.FlowImport(BASE)
flowImport.ratio = 2.0  # Larger canvas

flowImport.init_xys(xys, ids=layer_ids)

# Or manually override
spacemap.XYRANGE = 3000
spacemap.XYD = 15
```

### Working with Metadata

```python
import pandas as pd

# Load original data with metadata
df = pd.read_csv("cells.csv.gz")

# After registration, match aligned points with metadata
for i, slice_obj in enumerate(slices):
    layer_id = slice_obj.id

    # Get aligned points
    points_aligned = slice_obj.imgs[Slice.rawKey].get_points(Slice.align2Key)

    # Get original metadata
    layer_data = df[df['layer'] == layer_id]

    # Combine (assuming same order)
    layer_data['x_aligned'] = points_aligned[:, 0]
    layer_data['y_aligned'] = points_aligned[:, 1]
```

### Batch Processing Multiple Projects

```python
projects = ["project1", "project2", "project3"]

for project_name in projects:
    BASE = f"data/{project_name}"
    flowImport = spacemap.flow.FlowImport(BASE)

    if len(flowImport.slices) == 0:
        # New project - initialize
        flowImport.init_xys(xys_dict[project_name], ids_dict[project_name])

    # Process...
```

## TransformDB (Advanced)

For advanced transformation management:

```python
from spacemap.flow.outputs import TransformDB

# Create transformation database
transform_db = TransformDB()

# After registration, transformations are stored automatically
# Access them through the Slice objects
```

## Data Types and Formats

### Input Formats

**CSV with layer column:**
```csv
x,y,layer
100.5,200.3,0
150.2,180.7,0
...
```

**CODEX format:**
```csv
x,y,array,protein_1,protein_2,...
100.5,200.3,Z01,...
150.2,180.7,Z01,...
```

### Output Formats

**Aligned coordinates (CSV):**
```python
df_aligned.to_csv('aligned_cells.csv.gz',
                  index=False, compression='gzip')
```

**3D model (NPY):**
```python
np.save('model_3d.npy', points_3d)
```

## Best Practices

1. **Always backup your original data** - Space-map does this automatically in `{BASE}/raw/`

2. **Use consistent layer IDs** - String or integer, but be consistent

3. **Check parameters after initialization:**
   ```python
   print(f"XYRANGE: {spacemap.XYRANGE}, XYD: {spacemap.XYD}")
   ```

4. **Save intermediate results** - Transformations are saved automatically

5. **Document your workflow** - Keep notes of parameters used

## Troubleshooting

### Data Not Found

```python
# Check if data exists at a stage
try:
    points = slice_obj.imgs[Slice.rawKey].get_points(Slice.align2Key)
except FileNotFoundError:
    print("Data not available - run registration first")
```

### Memory Issues

```python
# Reduce resolution
spacemap.XYD = 20  # Larger value = less memory
```

### Coordinate Range Issues

```python
# Check if coordinates are within expected range
print(f"Data range should fit in: {spacemap.XYRANGE}")
```

## Examples

See complete examples in:
- [01_quickstart.ipynb](../../examples/01_quickstart.ipynb) - Basic data handling
- [02_advanced_registration.ipynb](../../examples/02_advanced_registration.ipynb) - Advanced techniques
- [Examples page](../examples/examples.md) - More scenarios

## Related Documentation

- [Key Workflows](../workflow/key-workflows.md) - Overall workflow
- [Registration System](../registration/registration-system-(lddmm).md) - Registration details
- [API Reference](../api/api.md) - Complete API documentation 