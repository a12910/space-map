# Grid Transformations

> Related source files:
> - [utils/fig.py](https://github.com/a12910/space-map/blob/ad208055/utils/fig.py)
> - [utils/grid.py](https://github.com/a12910/space-map/blob/ad208055/utils/grid.py)
> - [utils/grid3.py](https://github.com/a12910/space-map/blob/ad208055/utils/grid3.py)
> - [utils/grid_points.py](https://github.com/a12910/space-map/blob/ad208055/utils/grid_points.py)
> - [utils/imaris.py](https://github.com/a12910/space-map/blob/ad208055/utils/imaris.py)
> - [utils/output.py](https://github.com/a12910/space-map/blob/ad208055/utils/output.py)

## Introduction

Grid Transformation is a core mechanism in the SpaceMap framework for representing and applying nonlinear spatial transformations. It supports mapping points (such as cell coordinates) from original section space to registered 3D space, serving as the primary carrier structure for LDDMM registration results.

For more about LDDMM registration, see: [Registration System (LDDMM)](../registration/registration-system-(lddmm).md).

## Basic Concepts of Grid Transformations

### What is a Transformation Grid?

A transformation grid is a regular 2D array of displacement vectors that defines how each point in the source space maps to the target space. Each grid point contains an (x, y) vector for the target coordinates.

In SpaceMap, grids are represented as 3D tensors with shape `[N, N, 2]`, where N is the grid resolution and the last dimension represents (x, y) coordinates.

### Coordinate Systems

- **Grid Coordinates**: Normalized to [-1, 1]
- **Image/Point Coordinates**: Pixel or data coordinates, typically [0, N*xyd]
- The `xyd` (XY density) parameter defines the scaling relationship between them

## Core Operations

### Grid Sampling and Bilinear Interpolation

The core operation is sampling points through the grid (`grid_sample_points_vectorized`), implementing smooth point transformation:
1. Locate the grid cell containing the point
2. Calculate interpolation ratios
3. Perform bilinear interpolation using corner vectors
4. Scale the result to target coordinate system

**Interpolation Formula:**
```
result = (1-y_ratio) * [(1-x_ratio)*top_left + x_ratio*top_right] + 
         y_ratio * [(1-x_ratio)*bottom_left + x_ratio*bottom_right]
```

### Inverse Grid Generation

Inverse grids (`inverse_grid_train`) are obtained by minimizing the error between forward and inverse transformations, supporting bidirectional point mapping.

### Grid Generation from Point Sets

`points_gen_grid_train` can optimize and generate smooth grids based on point pair relationships, implementing nonlinear mapping between arbitrary point sets.

### Grid Merging

`merge_grid_img` supports sequential composition of multiple grid transformations, facilitating complex workflow chaining.

## Main Classes and Interfaces

- **GridGenerate**: Object-oriented interface supporting grid generation from point sets, grid refinement, and grid transformation application.
- **ModelGenerate**: Used for 3D model generation, applying grid transformations to cell boundaries and coordinates for registered 3D visualization.

## Performance and Parameters

- Grid resolution N determines precision and memory consumption
- `xyd` controls scaling between grid and image space
- Vectorized operations efficiently handle large-scale point sets
- Bilinear interpolation balances speed and accuracy
- Inverse grid computation is more time-consuming than forward transformation

## Integration with SpaceMap Workflow

- Grid transformations are fundamental to LDDMM registration, point set registration, and 3D reconstruction workflows
- All grid transformations between sections are uniformly managed through `TransformDB`
- Supports forward/inverse transformations, batch point set processing, and grid composition

---

> For reference source code and detailed implementation, please see the related source files above. 