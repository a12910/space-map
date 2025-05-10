# API Reference

This page provides an overview of SpaceMap's core modules and functions.

## Modules Overview

SpaceMap is organized into the following main modules:

| Module | Description |
|--------|-------------|
| `spacemap.flow` | Core flow-based registration system |
| `spacemap.registration` | Registration algorithms |
| `spacemap.affine` | Affine transformation utilities |
| `spacemap.matches` | Feature detection and matching algorithms |
| `spacemap.find` | Point detection and analysis tools |
| `spacemap.utils` | Utility functions for data processing and visualization |
| `spacemap.base` | Core base classes and functions |

## Slice

The `Slice` class is a fundamental component for handling tissue sections:

```python
from spacemap import Slice

# Key attributes
Slice.align1Key  # Key for first alignment
Slice.align2Key  # Key for second alignment

# Main methods
slice.get_transform_points()  # Get transformed points after registration
slice.get_points()  # Get original points
```

## flow

The `flow` module implements the main registration workflow.

```python
# Main flow classes for registration workflow
flow = spacemap.flow.FlowImport(base_dir)  # Initialize flow processing
flow.init_xys(xys, layer_ids)  # Set xy coordinates and layer IDs
slices = flow.slices  # Get slice objects

# Registration manager
mgr = spacemap.flow.AutoFlowMultiCenter3(slices)
mgr.alignMethod = "auto"  # Set alignment method
mgr.affine(method="DF", show=True)  # Perform affine registration
mgr.ldm_pair(Slice.align1Key, Slice.align2Key, show=True)  # Perform LDDMM registration

# Export results
export = spacemap.flow.FlowExport(slices)
# export.points_csv("output.csv")  # Export transformed points to CSV

# TransformDB for managing transformations
db = spacemap.flow.outputs.TransformDB()
db.add_transform(source_id, target_id, affine_transform, flow_transform)
db.save(filename)
db = spacemap.flow.outputs.TransformDB.load(filename)
```

## registration

The `registration` module provides functions for registering images and point sets.

```python
# Main registration functions
spacemap.registration.affine_register(source_points, target_points, **kwargs)
spacemap.registration.multimodal_register(source_points, target_points, source_image=None, target_image=None, **kwargs)
spacemap.registration.global_optimize(transform_db, **kwargs)
spacemap.registration.apply_transform_sequence(points, transform_db, source_id, target_id)
```

## affine

The `affine` module handles affine transformations.

```python
# Core affine functions
spacemap.affine.transform_points(points, affine_matrix)
spacemap.affine.transform_image(image, affine_matrix, output_shape=None)
spacemap.affine.estimate_affine(source_points, target_points, method='ransac')
```

## matches

The `matches` module provides feature matching algorithms.

```python
# Feature matching
keypoints1, keypoints2, matches = spacemap.matches.find_matches(image1, image2, method='sift')
spacemap.matches.filter_matches(keypoints1, keypoints2, matches, **kwargs)
```

## find

The `find` module contains tools for detecting and analyzing points.

```python
# Point detection and error analysis
keypoints = spacemap.find.detect_keypoints(image, method='sift')
errors = spacemap.find.err.calculate_registration_error(source_points, target_points, transformed_points)
```

## utils

The `utils` module provides various utility functions.

### utils.show

Visualization utilities.

```python
# Visualization functions
spacemap.utils.show.plot_image(image, **kwargs)
spacemap.utils.show.plot_points(points, **kwargs)
spacemap.utils.show.plot_matches(image1, image2, keypoints1, keypoints2, matches, **kwargs)
spacemap.utils.show.plot_transformation_field(transform, **kwargs)
spacemap.utils.show.interactive_3d_plot(points_3d, **kwargs)
```

### utils.img

Image processing utilities.

```python
# Image utilities
image = spacemap.utils.img.load_image(filename)
processed_image = spacemap.utils.img.preprocess_image(image, **kwargs)
spacemap.utils.img.save_image(image, filename)
```

### utils.compute

Computation utilities.

```python
# Computational utilities
normalized_points = spacemap.utils.compute.normalize_points(points)
distances = spacemap.utils.compute.pairwise_distances(points1, points2)
```

### utils.compare

Comparison utilities.

```python
# Comparison utilities
spacemap.utils.compare.interactive_comparison(source_points, target_points, aligned_points, **kwargs)
```

### utils.grid_points

Utilities for grid-based operations on point sets.

```python
# Grid operations
grid = spacemap.utils.grid_points.points_to_grid(points, grid_size)
interpolated_points = spacemap.utils.grid_points.interpolate_points(points, values, query_points)
```

### utils.imaris

Utilities for working with Imaris files.

```python
# Imaris utilities
spacemap.utils.imaris.export_points(points, filename, **kwargs)
```

## base

The `base` module contains core base classes and functionality.

```python
# Core settings
spacemap.base.root.DEVICE = 0  # Set to 'cpu' or GPU device number
spacemap.base.root.L  # Global logger

# Flow base
spacemap.base.flowBase.FlowBase  # Base class for flow transforms
```

## Complete Workflow Example

Here's an overview of a complete registration workflow:

```python
import spacemap
from spacemap import Slice
import pandas as pd

# 1. Load data
df = pd.read_csv("cells.csv.gz")
groups = df.groupby("layer")
xys = []  # xy coordinates
ids = []  # layer IDs

for layer, dff in groups:
    xy = dff[["x", "y"]].values
    ids.append(layer)
    xys.append(xy)

# 2. Initialize flow processing
base = "data/flow"
flow = spacemap.flow.FlowImport(base)
flow.init_xys(xys, ids)
slices = flow.slices

# 3. Perform registrations
mgr = spacemap.flow.AutoFlowMultiCenter3(slices)
mgr.alignMethod = "auto"
mgr.affine("DF", show=True)
mgr.ldm_pair(Slice.align1Key, Slice.align2Key, show=True)

# 4. Export results
export = spacemap.flow.FlowExport(slices)
```

## Examples

For complete usage examples, refer to:

- [Quick Start Guide](../index.md)
- [Usage Guide](../examples/usage.md)
- [Examples](../examples/examples.md)

## Detailed API Documentation

For detailed function signatures, parameters, and return types, refer to the auto-generated API documentation. 