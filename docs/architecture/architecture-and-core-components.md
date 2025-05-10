# Architecture and Core Components

> Related source files:
> - [__init__.py](https://github.com/a12910/space-map/blob/ad208055/__init__.py)
> - [base/base2.py](https://github.com/a12910/space-map/blob/ad208055/base/base2.py)
> - [base/root.py](https://github.com/a12910/space-map/blob/ad208055/base/root.py)
> - [flow/afFlowMultiCenter3.py](https://github.com/a12910/space-map/blob/ad208055/flow/afFlowMultiCenter3.py)
> - [flow/outputs.py](https://github.com/a12910/space-map/blob/ad208055/flow/outputs.py)
> - [docs/assets/images/logo.png](https://github.com/a12910/space-map/blob/ad208055/docs/assets/images/logo.png)
> - [docs/assets/images/qr.png](https://github.com/a12910/space-map/blob/ad208055/docs/assets/images/qr.png)
> - [docs/examples/examples.md](https://github.com/a12910/space-map/blob/ad208055/docs/examples/examples.md)
> - [docs/index.md](https://github.com/a12910/space-map/blob/ad208055/docs/index.md)
> - [docs/examples/usage.md](https://github.com/a12910/space-map/blob/ad208055/docs/examples/usage.md)
> - [mkdocs.yml](https://github.com/a12910/space-map/blob/ad208055/mkdocs.yml)

## Introduction

This page provides a high-level overview of the SpaceMap architecture and its main components. SpaceMap is designed as a modular framework focused on reconstructing 3D tissue maps from consecutive tissue sections, particularly suitable for spatial transcriptomics and proteomics data.

For detailed component descriptions, please refer to the sub-pages:
- [Registration System (LDDMM)](../registration/registration-system-(lddmm).md)
- [Grid Transformations](../grid/grid-transformations.md)
- [Data Management (Flow, Slice)](../data/data-management.md)
- [Feature Matching](../feature/feature-matching.md)

## System Architecture Overview

SpaceMap employs a layered architecture that decouples data management, registration, and visualization while ensuring efficient data flow.

## Core Components

SpaceMap consists of multiple interconnected components that work together to process and align tissue section data. The main classes and their relationships are as follows:

- **Registration System**: Core registration system, divided into:
  - Affine Registration: Global coarse alignment, handling translation, rotation, scaling, and shearing.
  - LDDMM Registration: Fine-grained nonlinear registration, preserving local structures.
- **Grid Transformations**: Fundamental mechanism for representing and applying nonlinear deformations, supporting forward/inverse transformations, bilinear interpolation, point cloud/image transformations, and transformation composition.
- **Data Management (Flow, Slice)**:
  - FlowImport: Loads raw data (cell coordinates, images), initializes the workflow.
  - Slice: Individual section object, containing raw data and all transformations.
  - TransformDB: Manages all transformations between sections.
  - FlowExport: Exports aligned sections and 3D models.
- **Feature Matching**:
  - SIFT: Traditional feature point matching.
  - LOFTR: Transformer-based deep learning feature matching.

## Data Processing Flow

The standard workflow is as follows:
1. FlowImport loads data
2. Slice represents each section
3. TransformDB manages all transformations
4. FlowExport exports results

## Multi-resolution Strategy

SpaceMap employs a multi-resolution registration strategy (e.g., AutoFlowMultiCenter3 class) to process at different resolutions, balancing efficiency and accuracy.

## TransformDB: Unified Transformation Management

TransformDB provides a unified interface supporting:
- Affine and grid transformations
- Forward/inverse transformations
- Automatic scaling to match input dimensions
- Batch processing of multiple images/point sets

## Configuration and Global Settings

Global configuration is defined in base/root.py, including:

| Setting      | Purpose                | Default Value         |
|--------------|------------------------|----------------------|
| XYRANGE      | Maximum coordinate range| 4000                 |
| XYD          | Coordinate transformation scale factor| 10           |
| BASE         | Data storage directory | "data/flow"          |
| DEVICE       | Computing device (CPU/GPU)| "cpu" or 0        |
| LAYER_START  | Starting layer index   | 0                    |
| LAYER_END    | Ending layer index     | 0                    |
| IMGCONF      | Image configuration    | {"raw": 1}           |

The `auto_init_xyd` and `init_xy` in base/base2.py can automatically initialize based on input data.

## Component Interaction

The diagram (original class/flow diagram, suggested to add images in docs directory) shows the interaction relationships between core components in a typical registration workflow.

## System Dependencies

- Requires Python and related scientific computing libraries (e.g., numpy, torch, opencv, kornia)
- Detailed dependencies can be found in the project's requirements.txt

---

> For reference source code and detailed implementation, please see the related source files above. 