# Key Workflows

> Related source files:
> - [docs/assets/images/logo.png](https://github.com/a12910/space-map/blob/ad208055/docs/assets/images/logo.png)
> - [docs/assets/images/qr.png](https://github.com/a12910/space-map/blob/ad208055/docs/assets/images/qr.png)
> - [docs/examples.md](https://github.com/a12910/space-map/blob/ad208055/docs/examples.md)
> - [docs/index.md](https://github.com/a12910/space-map/blob/ad208055/docs/index.md)
> - [docs/usage.md](https://github.com/a12910/space-map/blob/ad208055/docs/usage.md)
> - [flow/__init__.py](https://github.com/a12910/space-map/blob/ad208055/flow/__init__.py)
> - [flow/afFlow2MultiDF.py](https://github.com/a12910/space-map/blob/ad208055/flow/afFlow2MultiDF.py)
> - [mkdocs.yml](https://github.com/a12910/space-map/blob/ad208055/mkdocs.yml)

## Introduction

This page outlines the main processing workflows for 3D tissue reconstruction in the SpaceMap framework, covering the entire process from raw data import to 3D model generation and spatial analysis.

For detailed API, see [API Reference](../api/api.md), and for specific implementation cases, see [Examples and Use Cases](../examples/examples.md).

## Main Workflow Overview

SpaceMap employs a multi-stage workflow, combining multi-scale feature matching with LDDMM (Large Deformation Diffeomorphic Metric Mapping) to balance efficiency and accuracy.

**Main Processing Workflow:**
1. Data Import (FlowImport)
2. Image Registration (AutoFlowMultiCenter2/3)
3. 3D Reconstruction (TransformDB)
4. Spatial Analysis (FlowExport)

## Image Registration Workflow

- Coarse Registration: Affine registration + feature point matching (SIFT/LOFTR)
- Fine Registration: LDDMM nonlinear registration (ldm_pair), grid generation and merging
- Multi-center Registration Strategy: Using middle section as center, bidirectional registration to improve global consistency

## 3D Reconstruction Workflow

- Grid Storage: Each registration step result saved as grid
- TransformDB unified management of all transformations
- Apply grid transformations to sections for point set/image 3D reconstruction
- Support grid merging to ensure global consistency

## Spatial Transcriptomics Analysis Workflow

- Cell-gene expression matrix analysis
- Gene expression spatial distribution analysis
- Cell type identification and spatial statistics
- 3D visualization and export

## End-to-End Integration Example

Typical complete workflow:
1. FlowImport imports data
2. AutoFlowMultiCenter2/3 performs registration
3. TransformDB manages and applies all transformations
4. FlowExport exports 3D model and analysis results

## Key Parameters and Configuration

| Parameter      | Description              | Default    | Component           |
|----------------|--------------------------|------------|---------------------|
| alignMethod    | Initial registration method| "auto"    | AutoFlowMultiCenter2|
| gpu            | Computing GPU device     | None       | AutoFlowMultiCenter2|
| finalErr       | LDDMM convergence threshold| System default| ldm_pair        |
| centerTrain    | Center section training  | False      | ldm_pair            |
| show           | Show visualization results| False     | Multiple locations  |
| saveGridKey    | Grid save key           | pairGridKey| ldm_pair            |

## Programming and Development Guidelines

- Always initialize data with FlowImport
- Choose appropriate AutoFlow class (2/2DF/3)
- Execute affine, ldmm, grid merging in sequence
- Use TransformDB for unified transformation management
- Use FlowExport for 3D result export

---

> For reference source code and detailed implementation, please see the related source files above. 