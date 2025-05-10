# Registration System (LDDMM)

> Related source files:
> - [registration/__init__.py](https://github.com/a12910/space-map/blob/ad208055/registration/__init__.py)
> - [registration/lddmm.py](https://github.com/a12910/space-map/blob/ad208055/registration/lddmm.py)
> - [registration/lddmm2.py](https://github.com/a12910/space-map/blob/ad208055/registration/lddmm2.py)
> - [registration/ldm2/torch_LDDMM2D.py](https://github.com/a12910/space-map/blob/ad208055/registration/ldm2/torch_LDDMM2D.py)
> - [registration/ldm2/torch_LDDMMBase.py](https://github.com/a12910/space-map/blob/ad208055/registration/ldm2/torch_LDDMMBase.py)

## Introduction

The LDDMM (Large Deformation Diffeomorphic Metric Mapping) registration system is a core component of the SpaceMap framework, responsible for precise nonlinear alignment of images and point clouds. It achieves high-precision registration between tissue sections by computing smooth, topology-preserving transformations.

For more about grid transformation applications, see: [Grid Transformations](../grid/grid-transformations.md).

## Overview

The LDDMM registration system employs a multi-scale, two-stage registration strategy:
1. **Affine Registration**: Initial global linear transformation for coarse alignment
2. **Diffeomorphic Registration**: Smooth, invertible nonlinear transformation based on time-varying velocity fields for fine alignment

This approach captures complex local deformations while preserving the topological structure of the original data.

## System Architecture

The LDDMM registration system is primarily implemented through the `LDDMMRegistration` class, which inherits from the base `Registration` class and interfaces with the core LDDMM algorithm to complete mathematical optimization and transformation computation.

## Registration Process

The LDDMM registration process uses a hierarchical optimization strategy, aligning from coarse to fine to avoid local optima.

### Registration Stages

1. **Initial Affine Stage**: High v_scale and epsilon for rough alignment
2. **Intermediate Affine Stage**: Reduced parameters for improved accuracy
3. **Fine Affine Stage**: Further reduced parameters for precise alignment
4. **LDDMM Stage**: Switch to LDDMM mode for nonlinear transformation optimization
5. **Final LDDMM Stage**: Minimum epsilon for highest precision

#### Key Parameters Table

| Stage | Do Affine | Do LDDMM | v_scale | epsilon | niter  |
|-------|-----------|----------|---------|---------|--------|
| 1     | 1         | 0        | 8.0     | 10000   | 300    |
| 2     | 1         | 0        | 4.0     | 1000    | 1000   |
| 3     | 1         | 0        | 1.0     | 50      | 6000   |
| 4     | 0         | 1        | 1.0     | 1000    | 20000  |
| 5     | 0         | 1        | 1.0     | 1       | 20000  |

## Implementation Details

### LDDMMRegistration Class

- `run_affine()`: Execute affine registration only
- `run()`: Complete registration process
- `load_img(imgI, imgJ)`: Load registration images
- `load_params()/output_params()`: Parameter serialization and deserialization
- `apply_img(img)`: Apply transformation to new image
- `generate_img_grid()`: Generate transformation grid for visualization

### Transformation Structure

- **Affine Part**: 3×3 matrix (affineA), containing rotation, scaling, and translation
- **LDDMM Part**: Time-varying velocity field (vt0, vt1), defining nonlinear transformation

Parameter structure example:
```json
{
  "vt0": "velocity field x component",
  "vt1": "velocity field y component",
  "affineA": "3×3 affine matrix"
}
```

Parameters can be saved via `output_params_path()` and loaded via `load_params_path()`, enabling transformation persistence and reuse.

## Usage Examples

### Basic Registration

1. Create `LDDMMRegistration` instance
2. Load source and target images
3. Execute registration
4. Apply transformation or export parameters

### Affine Registration Only

If only linear alignment is needed, call `run_affine()`.

## Integration with SpaceMap

The LDDMM registration system is a core component of the SpaceMap alignment workflow, typically used after feature point coarse registration and before grid transformation. The final transformation is stored in `TransformDB` and can be used for subsequent processing of various data types.

---

> For reference source code and detailed implementation, please see the related source files above. 