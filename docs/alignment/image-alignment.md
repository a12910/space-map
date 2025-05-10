# Image Alignment

> Related source files:
> - [affine/AutoScale.py](https://github.com/a12910/space-map/blob/ad208055/affine/AutoScale.py)
> - [affine/BestRotate.py](https://github.com/a12910/space-map/blob/ad208055/affine/BestRotate.py)
> - [affine/FilterGlobal.py](https://github.com/a12910/space-map/blob/ad208055/affine/FilterGlobal.py)
> - [affine/FinalRotate.py](https://github.com/a12910/space-map/blob/ad208055/affine/FinalRotate.py)
> - [affine/ManualRotate.py](https://github.com/a12910/space-map/blob/ad208055/affine/ManualRotate.py)
> - [affine_block/AutoAffineImgKey.py](https://github.com/a12910/space-map/blob/ad208055/affine_block/AutoAffineImgKey.py)
> - [affine_block/FilterGlobalImg.py](https://github.com/a12910/space-map/blob/ad208055/affine_block/FilterGlobalImg.py)
> - [affine_block/ManualRotateImg.py](https://github.com/a12910/space-map/blob/ad208055/affine_block/ManualRotateImg.py)
> - [find/dice4.py](https://github.com/a12910/space-map/blob/ad208055/find/dice4.py)
> - [flow/__init__.py](https://github.com/a12910/space-map/blob/ad208055/flow/__init__.py)
> - [flow/afFlow2MultiDF.py](https://github.com/a12910/space-map/blob/ad208055/flow/afFlow2MultiDF.py)

## Introduction

The image alignment system in SpaceMap provides a multi-stage registration workflow for aligning consecutive tissue sections, enabling high-precision 3D reconstruction. This workflow combines affine registration (coarse alignment) and LDDMM (Large Deformation Diffeomorphic Metric Mapping, fine nonlinear deformation), balancing efficiency and accuracy.

- For 3D reconstruction details, see: [3D Reconstruction](../reconstruction/3d-reconstruction.md)
- For LDDMM registration system, see: [Registration System (LDDMM)](../registration/registration-system-(lddmm).md)

## Alignment Pipeline Overview

The registration workflow adopts a hierarchical coarse-to-fine strategy to optimize both efficiency and accuracy:

1. Preprocessing and normalization
2. Coarse registration (affine)
3. Fine registration (LDDMM)
4. Multi-center registration strategy
5. Transformation management and composition
6. Results used for 3D reconstruction

## Preprocessing and Normalization

Before registration, images need to be normalized for feature matching:
- Downsample to detect global intensity trends
- Calculate nonzero mean and normalize intensity
- Ensure consistent data format for subsequent processing

Related method: `AutoAffineImgKey.process_init`

## Coarse Registration (AutoAffineImgKey)

- Feature point matching (SIFT/LOFTR, auto selection)
- Match filtering (FilterGraphImg/FilterGlobalImg)
- Multi-resolution affine optimization (LDMAffine, AutoGradImg)

Related code snippets:
- [AutoAffineImgKey.py#L6-L68](https://github.com/a12910/space-map/blob/ad208055/affine_block/AutoAffineImgKey.py#L6-L68)
- [FilterGlobalImg.py#L6-L51](https://github.com/a12910/space-map/blob/ad208055/affine_block/FilterGlobalImg.py#L6-L51)

## Fine Registration (LDDMM)

- Input coarse registration results
- Initialize LDDMMRegistration, set GPU/error threshold and other parameters
- Compute nonlinear deformation, generate grid
- Store grid in TransformDB

Related code snippets:
- [afFlow2MultiDF.py#L21-L60](https://github.com/a12910/space-map/blob/ad208055/flow/afFlow2MultiDF.py#L21-L60)
- [afFlow2MultiDF.py#L128-L172](https://github.com/a12910/space-map/blob/ad208055/flow/afFlow2MultiDF.py#L128-L172)

## Multi-center Registration Strategy

- Select the middle section as the center, perform bidirectional registration
- Parallel processing to improve efficiency
- Merge transformations to ensure global consistency

Related code snippets:
- [afFlow2MultiDF.py#L62-L89](https://github.com/a12910/space-map/blob/ad208055/flow/afFlow2MultiDF.py#L62-L89)
- [afFlow2MultiDF.py#L74-L126](https://github.com/a12910/space-map/blob/ad208055/flow/afFlow2MultiDF.py#L74-L126)

## Transformation Management and Composition

- Transformations are stored as grids in TransformDB
- Supports grid composition and application to images/point sets

Related code snippets:
- [afFlow2MultiDF.py#L128-L172](https://github.com/a12910/space-map/blob/ad208055/flow/afFlow2MultiDF.py#L128-L172)

## Fine-tuning and Manual Intervention

| Component     | Purpose                              | Implementation Method                  |
|--------------|--------------------------------------|----------------------------------------|
| AutoScale    | Detect/correct scale differences      | Region of interest detection and scale calculation |
| BestRotate   | Coarse/fine rotation search           | Multi-step coarse-to-fine rotation search         |
| FinalRotate  | Fine rotation adjustment              | Sliding window to find optimal rotation           |
| ManualRotate | Manual rotation/scale/translation     | Provides interactive interface                    |

Related code snippets:
- [AutoScale.py#L5-L26](https://github.com/a12910/space-map/blob/ad208055/affine/AutoScale.py#L5-L26)
- [BestRotate.py#L5-L66](https://github.com/a12910/space-map/blob/ad208055/affine/BestRotate.py#L5-L66)
- [FinalRotate.py#L5-L36](https://github.com/a12910/space-map/blob/ad208055/affine/FinalRotate.py#L5-L36)
- [ManualRotate.py#L6-L63](https://github.com/a12910/space-map/blob/ad208055/affine/ManualRotate.py#L6-L63)

## Integration with 3D Reconstruction

- Registration results are directly used for 3D reconstruction
- Supports section stacking, interpolation, and spatial analysis

Related code snippets:
- [flow/__init__.py#L8-L14](https://github.com/a12910/space-map/blob/ad208055/flow/__init__.py#L8-L14)

## Summary

The image alignment system in SpaceMap provides a solid foundation for 3D tissue reconstruction and spatial transcriptomics analysis through a multi-stage coarse-to-fine workflow, combining affine and LDDMM techniques, robust feature point matching and filtering, grid transformation management, and multi-center parallel strategies. 