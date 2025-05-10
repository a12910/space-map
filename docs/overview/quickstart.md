# Quick Start Guide

This guide will help you get started with SpaceMap by walking through a basic example.

![Workflow Diagram](../assets/images/workflow.png)

## Prerequisites

Before starting, make sure you have:

1. Installed SpaceMap (see [Installation Guide](installation.md))
2. Prepared your spatial data (e.g., cell coordinates, images)

## Basic Usage

### Importing SpaceMap

```python
import spacemap
from spacemap import Slice
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

### Loading and Processing Cell Coordinate Data

```python
# Load cell coordinate data from CSV file
# CSV file should contain cell coordinates and layer information
df = pd.read_csv("path/to/cells.csv.gz")
groups = df.groupby("layer")

# Organize data by layers
xys = []  # xy coordinates (list of N*2 numpy arrays)
ids = []  # layer IDs

for layer, dff in groups:
    xy = dff[["x", "y"]].values
    ids.append(layer)
    xys.append(xy)

# Set project base folder for storing all data
base = "data/flow"

# Initialize flow processing
flow = spacemap.flow.FlowImport(base)

# Initialize with xy coordinates and layer IDs
flow.init_xys(xys, ids)

# Get slice objects
slices = flow.slices
```

### Performing Affine Registration

SpaceMap uses a two-stage registration approach. First, we perform affine registration for coarse alignment:

```python
# Create registration manager
mgr = spacemap.flow.AutoFlowMultiCenter3(slices)

# Set alignment method
mgr.alignMethod = "auto"

# Perform affine registration
# The "DF" parameter specifies the method
# show=True will display visualization of the registration process
mgr.affine("DF", show=True)
```

### Fine Registration with LDDMM

After affine registration, we can perform fine registration using LDDMM:

```python
# Perform LDDMM registration between adjacent sections
# This provides high-precision alignment while preserving local structures
mgr.ldm_pair(Slice.align1Key, Slice.align2Key, show=True)
```

### Exporting Results

```python
# Export registration results
export = spacemap.flow.FlowExport(slices)
imgs = export.export_imgs()
points = export.

# Export can be used to save transformed coordinates
# export.points_csv("aligned_cells.csv")
```

## Next Steps

After completing this quick start guide, you can explore more advanced features:

- [Detailed Usage Guide](../examples/usage.md) for more complex examples
- [API Reference](../api/api.md) for comprehensive documentation
- [Examples](../examples/examples.md) for specific use cases

For any issues or questions, please [open an issue](https://github.com/a12910/spacemap/issues) on GitHub. 