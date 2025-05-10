# Spatial Transcriptomics Analysis

> Related source files:
> - [find/__init__.py](https://github.com/a12910/space-map/blob/ad208055/find/__init__.py)
> - [find/basic.py](https://github.com/a12910/space-map/blob/ad208055/find/basic.py)
> - [find/cellShape.py](https://github.com/a12910/space-map/blob/ad208055/find/cellShape.py)
> - [find/nearBound.py](https://github.com/a12910/space-map/blob/ad208055/find/nearBound.py)
> - [find/nearBoundCellData.py](https://github.com/a12910/space-map/blob/ad208055/find/nearBoundCellData.py)
> - [find/nearBoundThread.py](https://github.com/a12910/space-map/blob/ad208055/find/nearBoundThread.py)
> - [flow/afFlow2Multi.py](https://github.com/a12910/space-map/blob/ad208055/flow/afFlow2Multi.py)
> - [utils/compare/cell_metrix.py](https://github.com/a12910/space-map/blob/ad208055/utils/compare/cell_metrix.py)
> - [utils/compare/filt.py](https://github.com/a12910/space-map/blob/ad208055/utils/compare/filt.py)
> - [utils/he_img.py](https://github.com/a12910/space-map/blob/ad208055/utils/he_img.py)

## Purpose and Scope

This section documents the tools and methodologies for analyzing spatial transcriptomics data within the SpaceMap framework. The analysis capabilities enable researchers to quantify, compare, and validate gene expression patterns across aligned tissue sections. For information about the alignment process itself, see [Image Alignment](../alignment/image-alignment.md), and for the 3D reconstruction process, see [3D Reconstruction](../reconstruction/3d-reconstruction.md).

## Overview of Spatial Transcriptomics Analysis Components

- Expression analysis: grid-based quantification and normalization
- Layer distance calculation: quantifies similarity between adjacent sections
- Parallelized processing: efficient computation for large datasets
- Cell boundary and shape analysis: validates registration quality
- Cell type distribution analysis: compares cell type patterns across sections

## Cell Expression Analysis

The cell expression analysis module processes spatial transcriptomics data to compute metrics that quantify gene expression patterns across tissue sections. The core of this functionality is implemented in the `cell_metrix.py` module.

### Grid-Based Expression Processing

Expression data is analyzed by dividing the tissue space into a grid and aggregating cell expression values within each grid cell:

- Divide the space into a grid with configurable size and overlap
- Sum expression values within each grid cell
- Normalize the expression values within each grid cell

See: [`cell_metrix.py`](https://github.com/a12910/space-map/blob/ad208055/utils/compare/cell_metrix.py#L5-L33)

### Layer Distance Calculation

The `calculate_layer_distances` function computes expression similarity between adjacent tissue sections:

- For each layer pair, compare grid-processed expression matrices
- Calculate absolute differences between corresponding grid cells
- Normalize by the number of non-empty grid cells
- Mean distance across all layer pairs provides an alignment quality metric

See: [`cell_metrix.py`](https://github.com/a12910/space-map/blob/ad208055/utils/compare/cell_metrix.py#L50-L91)

### Parallelized Processing

For large datasets, the framework provides `caculate_layer_distances_thread` for multi-threaded execution:

- Create parameter sets for each comparison (datasets, grid sizes, overlaps)
- Launch parallel workers using Python's multiprocessing
- Collect and aggregate results from all workers

See: [`cell_metrix.py`](https://github.com/a12910/space-map/blob/ad208055/utils/compare/cell_metrix.py#L93-L118)

## Cell Boundary and Shape Analysis

SpaceMap provides tools to analyze cell boundaries and shapes across aligned tissue sections, enabling validation of registration quality through cell morphology.

### Nearest Cell Boundary Analysis

The `NearBoundGenerate` class identifies corresponding cells between adjacent sections and computes boundary-based similarity metrics:

- Apply spatial transformations to cell boundaries using TransformDB
- Identify nearest neighboring cells between adjacent layers
- Compute boundary similarity metrics between matched cells
- Aggregate and export metrics for quality assessment

See: [`nearBound.py`](https://github.com/a12910/space-map/blob/ad208055/find/nearBound.py#L11-L246)

### Cell Shape Analysis

The `CellShapeGenerate` class extends boundary analysis with shape-specific metrics:

- Compare bounding box dimensions (min/max coordinates)
- Calculate aspect ratios before and after transformation
- Compute area preservation metrics
- Quantify overall shape similarity

See: [`cellShape.py`](https://github.com/a12910/space-map/blob/ad208055/find/cellShape.py#L8-L84)

## Cell Type Distribution Analysis

SpaceMap provides tools to analyze the distribution of cell types across aligned tissue sections. This functionality helps validate alignment quality by ensuring similar cell type distributions in adjacent regions.

### Cell Type Comparison Methods

- `cmp_layers_label`: Compares cell distributions by cell type labels
- `cmp_filter_part`: Filters cells within a specific spatial region
- `cmp_adjacent_layers`: Compares cell distributions in adjacent layers
- `compare_workflow`: Orchestrates comparison between different datasets

See: [`filt.py`](https://github.com/a12910/space-map/blob/ad208055/utils/compare/filt.py#L11-L159)

## Integration with Registration Pipeline

The spatial transcriptomics analysis components are integrated with the broader registration pipeline through the `AutoFlowMultiCenter2` class, which coordinates the alignment of tissue sections:

- Transformations computed by the registration pipeline are stored in TransformDB
- Spatial transcriptomics analysis uses these transformations via the TransformDB API
- Quality metrics from the analysis can be used to refine registration parameters
- The multi-center approach in `AutoFlowMultiCenter2` starts from a center slice and works outward in both directions

See: [`afFlow2Multi.py`](https://github.com/a12910/space-map/blob/ad208055/flow/afFlow2Multi.py#L7-L157)

## Practical Usage

### Cell Expression Matrix Analysis

To calculate distances between expression patterns in adjacent layers:

```python
# Example usage for calculating layer distances
from spacemap.utils.compare.cell_metrix import calculate_layer_distances
# ... load your expression data ...
distances = calculate_layer_distances(expression_matrices)
```

For large datasets, use the parallelized version:

```python
from spacemap.utils.compare.cell_metrix import caculate_layer_distances_thread
# ... load your expression data ...
results = caculate_layer_distances_thread(expression_matrices)
```

### Cell Boundary Analysis

To analyze cell boundaries across sections:

```python
from spacemap.find.nearBound import NearBoundGenerate
# ... prepare your cell boundary data ...
nbg = NearBoundGenerate(...)
nbg.run()
```

### Cell Type Distribution Comparison

To compare cell type distributions across sections:

```python
from spacemap.utils.compare.filt import cmp_layers_label
# ... prepare your cell type data ...
result = cmp_layers_label(layer1, layer2)
```

---

> For reference source code and detailed implementation, please see the related source files above. 