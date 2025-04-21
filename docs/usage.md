# Detailed Usage Guide

This guide provides detailed examples for using SpaceMap's core functionalities.

## Table of Contents

- [Data Preparation](#data-preparation)
- [Registration Workflow](#registration-workflow)
- [Advanced Registration](#advanced-registration)
- [Working with Large Datasets](#working-with-large-datasets)
- [Visualizing Results](#visualizing-results)
- [Exporting 3D Models](#exporting-3d-models)

## Data Preparation

### Supported Data Formats

SpaceMap works with various data types:

```python
import spacemap as sm
import numpy as np
import pandas as pd
import tifffile

# Cell coordinates (numpy array)
points = np.array([[x1, y1], [x2, y2], ...])  # Shape: (n_cells, 2)

# Cell coordinates with metadata (pandas DataFrame)
cells_df = pd.DataFrame({
    'x': [x1, x2, ...],
    'y': [y1, y2, ...],
    'cell_type': ['T-cell', 'B-cell', ...],
    'gene_expression': [0.5, 0.7, ...]
})

# Image data (numpy array)
img = tifffile.imread('section_image.tif')  # Shape: (height, width) or (height, width, channels)
```

### Data Preprocessing

```python
# Filter cells by type
t_cells = cells_df[cells_df['cell_type'] == 'T-cell']
t_cell_coords = t_cells[['x', 'y']].values

# Normalize coordinates
normalized_points = sm.utils.compute.normalize_points(points)

# Process image data
processed_img = sm.utils.img.preprocess_image(img, contrast_enhance=True, denoise=True)
```

## Registration Workflow

### Full Registration Pipeline

```python
# Load data for consecutive sections
sections_data = []
for i in range(5):  # 5 sections
    # Load points and images
    points = np.load(f"section{i}_points.npy")
    img = tifffile.imread(f"section{i}_image.tif")
    sections_data.append({"points": points, "image": img})

# Initialize transformation database
transform_db = sm.flow.outputs.TransformDB()

# Register consecutive sections
for i in range(len(sections_data) - 1):
    source = sections_data[i]
    target = sections_data[i+1]
    
    # Step 1: Affine registration
    affine_transform = sm.registration.affine_register(
        source_points=source["points"],
        target_points=target["points"]
    )
    
    # Apply affine transformation
    source_points_affine = sm.affine.transform_points(source["points"], affine_transform)
    
    # Step 2: Feature matching (optional, for improved alignment)
    keypoints_source, keypoints_target, matches = sm.matches.find_matches(
        source["image"], target["image"], method="loftr"
    )
    
    # Step 3: Fine registration with flow
    flow_transform = sm.flow.register(
        source_points=source_points_affine,
        target_points=target["points"],
        source_keypoints=keypoints_source,
        target_keypoints=keypoints_target,
        matches=matches,
        iterations=200
    )
    
    # Store transforms in database
    transform_db.add_transform(
        source_id=i,
        target_id=i+1,
        affine_transform=affine_transform,
        flow_transform=flow_transform
    )

# Save transformation database
transform_db.save("registration_results.pkl")
```

## Advanced Registration

### Multi-modal Registration

```python
# Register using both point coordinates and image features
result = sm.registration.multimodal_register(
    source_points=source_points,
    target_points=target_points,
    source_image=source_image,
    target_image=target_image,
    point_weight=0.7,  # Weight for point-based alignment
    image_weight=0.3   # Weight for image-based alignment
)
```

### Global Consistency Optimization

```python
# Load transformation database
transform_db = sm.flow.outputs.TransformDB.load("registration_results.pkl")

# Optimize for global consistency
optimized_transforms = sm.registration.global_optimize(transform_db)

# Apply optimized transformations to all sections
aligned_sections = []
for i in range(len(sections_data)):
    if i == 0:  # Reference section
        aligned_sections.append(sections_data[0]["points"])
    else:
        # Apply sequence of transformations to align to reference
        aligned_points = sm.registration.apply_transform_sequence(
            sections_data[i]["points"],
            optimized_transforms,
            source_id=i,
            target_id=0
        )
        aligned_sections.append(aligned_points)
```

## Working with Large Datasets

### Chunked Processing

```python
# Process large point clouds in chunks
chunk_size = 10000  # Number of points per chunk
num_chunks = len(large_point_set) // chunk_size + 1

transformed_points = []
for i in range(num_chunks):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, len(large_point_set))
    
    chunk = large_point_set[start_idx:end_idx]
    transformed_chunk = sm.flow.transform_points(chunk, flow_transform)
    transformed_points.append(transformed_chunk)

# Combine all transformed chunks
transformed_points = np.vstack(transformed_points)
```

### GPU Acceleration

```python
# Enable GPU acceleration
sm.base.root.DEVICE = 0  # Use first GPU

# Perform registration with GPU acceleration
flow_transform = sm.flow.register(
    source_points=source_points,
    target_points=target_points,
    use_gpu=True
)
```

## Visualizing Results

### Interactive Visualization

```python
# Compare before/after registration
sm.utils.compare.interactive_comparison(
    source_points=source_points,
    target_points=target_points,
    aligned_points=transformed_points
)

# Visualize transformation field
sm.utils.show.plot_transformation_field(flow_transform)

# Interactive 3D model visualization
sm.utils.show.interactive_3d_plot(model_3d)
```

### Custom Visualization

```python
# Create custom visualization with cell types
cell_types = ['T-cell', 'B-cell', 'Epithelial', 'Stromal']
cell_colors = ['red', 'blue', 'green', 'purple']

# Plot cells colored by type
plt.figure(figsize=(10, 8))
for cell_type, color in zip(cell_types, cell_colors):
    mask = cells_df['cell_type'] == cell_type
    plt.scatter(
        cells_df.loc[mask, 'x'], 
        cells_df.loc[mask, 'y'],
        c=color,
        label=cell_type,
        s=3,
        alpha=0.7
    )
plt.legend()
plt.title('Cells by Type')
plt.show()
```

## Exporting 3D Models

### Exporting to Common Formats

```python
# Export 3D model to OBJ format
sm.utils.output.export_to_obj(model_3d, "tissue_model.obj")

# Export as point cloud
sm.utils.output.export_point_cloud(model_3d, "tissue_model.ply")

# Export to Imaris format
sm.utils.imaris.export_points(model_3d, "tissue_model.ims", point_size=5)
```

### Saving Results for Analysis

```python
# Save aligned data for further analysis
np.save("aligned_sections.npy", aligned_sections)

# Export as CSV with metadata
aligned_df = pd.DataFrame({
    'x': model_3d[:, 0],
    'y': model_3d[:, 1],
    'z': model_3d[:, 2],
    'section': section_indices,
    'cell_type': cell_types
})
aligned_df.to_csv("aligned_model.csv", index=False)
```

## Advanced Topics

For more advanced usage, please refer to:

- [API Reference](api.md) for detailed function documentation
- [Examples](examples.md) for specific use cases
- Individual module documentation for specialized features 