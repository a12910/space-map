# Architecture and Core Components

This section provides a high-level overview of the SpaceMap architecture and its primary components. SpaceMap is designed as a modular framework for reconstructing 3D tissue maps from serial tissue sections, with a particular focus on spatial transcriptomics and proteomics data.

For detailed information about specific components, see the following sub-sections:
- [Registration System (LDDMM)](../registration/registration-system-(lddmm).md)
- [Grid Transformations](../grid/grid-transformations.md)
- [Data Management (Flow, Slice)](../data/data-management.md)
- [Feature Matching](../feature/feature-matching.md)

## System Architecture Overview

SpaceMap employs a layered architecture that separates data management, registration, and visualization concerns while maintaining efficient data flow between components.

## Core Components

SpaceMap consists of several interconnected components that work together to process and align tissue section data:
- **FlowImport**: Responsible for loading raw data (cell coordinates, images) and initializing the processing pipeline.
- **Slice**: Represents a single tissue section, containing both the raw data and any transformations applied to it.
- **TransformDB**: Stores and manages transformations between different sections.
- **FlowExport**: Handles exporting of aligned sections and 3D models.

## Registration System

The registration system is the core of SpaceMap, responsible for aligning tissue sections. It consists of two primary components:
1. **Affine Registration**: Provides coarse alignment by computing global transformations that account for translation, rotation, scaling, and shearing.
2. **LDDMM Registration**: Performs fine-grained, non-linear registration that preserves local structures while accurately aligning sections.

The two-stage approach combines computational efficiency with high accuracy, allowing SpaceMap to handle large datasets while maintaining precision.

For more detailed information about the registration process, see the [Registration System](../registration/registration-system-(lddmm).md) documentation.

## Grid Transformations

Grid transformations are a fundamental mechanism in SpaceMap for representing and applying non-linear deformations. These transformations are represented as regular grids where each point maps to another location, enabling complex warping operations.

Key features:
- Supports both forward and inverse transformations
- Provides bilinear interpolation for points between grid vertices
- Can be applied to both images and point clouds
- Facilitates combination of multiple transformations

## Data Management (Flow, Slice)

The data management components in SpaceMap handle the import, manipulation, and export of tissue section data. The primary classes are:
- **FlowImport**: Responsible for loading raw data (cell coordinates, images) and initializing the processing pipeline.
- **Slice**: Represents a single tissue section, containing both the raw data and any transformations applied to it.
- **TransformDB**: Stores and manages transformations between different sections.
- **FlowExport**: Handles exporting of aligned sections and 3D models.

For information about data management and preprocessing, refer to the [Data Management](../data/data-management.md) guide.

## Feature Matching

Feature matching components are used to find correspondences between different tissue sections, which are then used as constraints in the registration process:
- **SIFT Matching**: Traditional computer vision approach for matching distinctive points between images.
- **LOFTR (Learning Optical Flow from Transformers)**: Deep learning-based method for feature matching, particularly effective for histological images.

## Multi-Resolution Strategy

SpaceMap employs a multi-resolution approach to registration to balance efficiency and accuracy. The `AutoFlowMultiCenter3` class in particular implements this strategy, enabling processing at multiple resolution levels for optimal results.

## TransformDB: Unified Transformation Management

The `TransformDB` class plays a central role in SpaceMap by providing a unified interface for storing and applying transformations:
- Supports both affine and grid-based transformations
- Forward and inverse transformations
- Automatic scaling to match input dimensions
- Batch processing for multiple images or point sets

## Configuration and Global Settings

SpaceMap uses global configuration settings to control various aspects of the system. These are defined in `base/root.py` and include:

| Setting      | Purpose                              | Default Value         |
|--------------|--------------------------------------|----------------------|
| XYRANGE      | Maximum coordinate range             | 4000                 |
| XYD          | Scale factor for coordinate transforms| 10                   |
| BASE         | Base directory for data storage      | "data/flow"          |
| DEVICE       | Computing device (CPU or GPU index)  | "cpu" or 0           |
| LAYER_START  | Starting layer index                 | 0                    |
| LAYER_END    | Ending layer index                   | 0                    |
| IMGCONF      | Image configuration settings         | {"raw": 1}           |

The `auto_init_xyd` and `init_xy` functions in `base/base2.py` provide convenient ways to initialize these settings based on input data.

# Key Workflows

This section outlines the primary processing workflows in the SpaceMap framework for 3D tissue reconstruction. It provides a high-level overview of the main processing pipelines, their components, and how they interact. The workflows described here cover the entire process from raw data import to final 3D model generation and spatial analysis.

## Main Workflows Overview

SpaceMap implements a multi-stage approach to 3D tissue reconstruction, combining multi-scale feature matching with large-deformation diffeomorphic metric mapping (LDDMM). The core workflows are designed to balance computational efficiency with registration accuracy.

**Main Processing Pipeline:**

1. **Data Import** – Import and organize serial section data using `FlowImport`.
2. **Image Alignment** – Multi-stage registration process to align neighboring sections using `AutoFlowMultiCenter2/3`.
3. **3D Reconstruction** – Build 3D tissue models from aligned sections using `TransformDB`.
4. **Spatial Analysis** – Analyze spatial transcriptomics data in the 3D context using `FlowExport`.

## Image Alignment Workflow

The image alignment workflow is a core component of SpaceMap, combining coarse affine registration with fine LDDMM registration to accurately align tissue sections. This two-stage approach preserves both global structure and local micro-anatomy.

**Image Alignment Pipeline:**

1. **Data Import:** Raw data (cell coordinates and optional images) is imported using `FlowImport.init_xys()`, which organizes the data for processing.
2. **Slice Creation:** The imported data is organized into `Slice` objects, which serve as the basic data structure for the alignment process.
3. **Coarse Alignment:**
   - **Affine Registration:** Initial alignment using `AutoFlowMultiCenter2.affine()` to handle large displacements between sections.
   - **Feature Matching:** Extraction of corresponding features between sections using SIFT or LOFTR for improved alignment.
4. **Fine Alignment:**
   - **LDDMM Registration:** Local non-linear alignment using `AutoFlowMultiCenter2.ldm_pair()` or `AutoFlowMultiCenter2DF.ldm_pair()`.
   - **Grid Generation:** Creation of transformation grids using `LDDMMRegistration.generate_img_grid()`.
   - **Grid Merging:** Combination of transformation grids using `ldm_merge_pair()` for global consistency.

The `AutoFlowMultiCenter2DF` class implements a multi-center approach that handles bidirectional alignment from a central slice, improving global consistency in the alignment.

## 3D Reconstruction Workflow

The 3D reconstruction workflow transforms aligned 2D sections into a coherent 3D model by computing transformation grids and applying them to the original data.

**3D Reconstruction Pipeline:**

1. **Grid Processing:**
   - **Grid Saving:** Each alignment transformation is saved as a grid using `Slice.data.saveGrid()`.
   - **TransformDB Creation:** A `TransformDB` object is created to store and manage all transformation grids.
   - **Grid Application:** Transformation grids are applied to sections with `_apply_grid()`.
2. **3D Model Building:**
   - **Point Transformation:** Cell coordinates are transformed using `TransformDB.apply_points()`.
   - **Model Export:** The 3D model is exported with `FlowExport.export_imgs()` for visualization and analysis.

The grid merging process is a key component of the 3D reconstruction workflow, implemented in `AutoFlowMultiCenter2DF.ldm_merge_pair()`. This method ensures that transformation grids are properly merged to maintain global consistency in the 3D reconstruction.

## Spatial Transcriptomics Analysis Workflow

The spatial transcriptomics analysis workflow enables analysis of gene expression patterns in the 3D reconstructed tissue.

**Spatial Analysis Pipeline:**

1. **Data Analysis:**
   - **Cell Matrix Analysis:** Processing of cell-gene expression matrices within the 3D context.
   - **Gene Expression Analysis:** Analysis of gene expression patterns across the 3D tissue.
   - **Cell Type Identification:** Classification of cells based on gene expression profiles.
2. **Visualization & Export:**
   - **Spatial Statistics:** Calculation of spatial relationships and patterns among cell types.
   - **3D Visualization:** Interactive visualization of cells with gene expression data in 3D space.

The `FlowExport` class handles the export of 3D models and transformed cell coordinates for downstream analysis.

## Multi-center Workflow Strategy

SpaceMap implements special multi-center workflows through the `AutoFlowMultiCenter2DF` and `AutoFlowMultiCenter3` classes. These workflows start from a central slice and work outward in both directions, which helps maintain global consistency in the reconstruction.

**Multi-center Alignment Strategy:**

1. **Center Slice Selection:** The middle slice is selected as a reference point.
2. **Bidirectional Processing:** Slices are processed in both forward and backward directions from the center.
3. **Parallel Computation:** Slice pairs can be processed in parallel for efficiency.
4. **Grid Merging:** Transformation grids are merged to ensure global consistency.

This bidirectional processing strategy helps maintain the global structure of the tissue while allowing for local deformations.

## Configuration Options and Parameters

The SpaceMap workflows can be customized through various configuration options and parameters:

| Parameter      | Description                          | Default         | Component                |
|---------------|--------------------------------------|-----------------|--------------------------|
| alignMethod   | Method for initial alignment         | "auto"          | AutoFlowMultiCenter2     |
| gpu           | GPU device to use for computation    | None            | AutoFlowMultiCenter2     |
| finalErr      | Error threshold for LDDMM convergence| System default  | ldm_pair                 |
| centerTrain   | Whether to use center-based training | False           | ldm_pair                 |
| show          | Whether to display visual results    | False           | Various methods          |
| saveGridKey   | Key for saving transformation grids  | pairGridKey     | ldm_pair                 |

These parameters allow fine-tuning of the workflows for different types of data and reconstruction requirements.

## Programming Against the Workflow System

When developing with the SpaceMap framework, it's important to understand the main workflow components and how they work together. The following best practices are recommended:

1. **Initialize with FlowImport:** Always start by creating a `FlowImport` object to organize your data.
2. **Use the appropriate AutoFlow class:** Select between `AutoFlowMultiCenter2`, `AutoFlowMultiCenter2DF`, or `AutoFlowMultiCenter3` depending on your alignment needs.
3. **Follow the workflow steps sequentially:** First perform affine alignment, then LDDMM, then merge the grids.
4. **Use TransformDB for transformation management:** Store all transformations in a `TransformDB` for efficient application.
5. **Export results with FlowExport:** Use `FlowExport` to save your 3D models and transformed data.

# Detailed Usage Guide

This guide provides detailed examples for using Space-map's core functionalities. Space-map employs a sophisticated two-stage registration approach that combines dimensionality reduction with robust feature matching to handle large-scale single-cell data efficiently.

## Table of Contents

- [Data Preparation](#data-preparation)
- [Registration Workflow](#registration-workflow)
- [Advanced Registration](#advanced-registration)
- [Working with Large Datasets](#working-with-large-datasets)
- [Visualizing Results](#visualizing-results)
- [Exporting 3D Models](#exporting-3d-models)

## Data Preparation

### Supported Data Formats

Space-map works with various data types:

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

- [API Reference](../api/api.md) for detailed function documentation
- [Examples](examples.md) for specific use cases
- Individual module documentation for specialized features 