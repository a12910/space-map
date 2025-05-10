# Feature Matching

> Related source files:
> - [affine/AutoGrad.py](https://github.com/a12910/space-map/blob/ad208055/affine/AutoGrad.py)
> - [affine_block/LdmAffine.py](https://github.com/a12910/space-map/blob/ad208055/affine_block/LdmAffine.py)
> - [base/flowBase.py](https://github.com/a12910/space-map/blob/ad208055/base/flowBase.py)
> - [flow/flowImport.py](https://github.com/a12910/space-map/blob/ad208055/flow/flowImport.py)
> - [matches/LOFTR.py](https://github.com/a12910/space-map/blob/ad208055/matches/LOFTR.py)
> - [matches/LOFTR2.py](https://github.com/a12910/space-map/blob/ad208055/matches/LOFTR2.py)
> - [matches/matchShow.py](https://github.com/a12910/space-map/blob/ad208055/matches/matchShow.py)
> - [matches/siftFind.py](https://github.com/a12910/space-map/blob/ad208055/matches/siftFind.py)
> - [registration/ldm/torch_LDDMMBase.py](https://github.com/a12910/space-map/blob/ad208055/registration/ldm/torch_LDDMMBase.py)

## Introduction

Feature matching is a key component in the SpaceMap framework, used to identify corresponding points between images for accurate registration of tissue sections. SpaceMap implements two main feature matching methods: SIFT (Scale-Invariant Feature Transform) and LOFTR (Local Feature Transformer).

For more information about the registration workflow, see: [Registration System (LDDMM)](../registration/registration-system-(lddmm).md) and [Image Alignment](../alignment/image-alignment.md).

## Overview

The feature matching system is mainly responsible for:
1. Detecting salient features in source and target images
2. Finding correspondences between these features
3. Filtering outlier matches
4. Providing matches to the registration algorithm for alignment

## Feature Matching Components

SpaceMap implements two main feature matching methods:
- SIFT (traditional computer vision method)
- LOFTR (deep learning-based feature matching)

### SIFT Matching

SIFT (Scale-Invariant Feature Transform) is a traditional local feature detection and description algorithm. In SpaceMap, SIFT is the default feature matching method.

#### Key Functions
1. **Feature Detection**: `__sift_kp()` detects keypoints and computes descriptors.
2. **Feature Matching**: `siftImageAlignment()` matches features between images, applies ratio test to filter matches, and returns corresponding point lists.

#### Implementation Details
- Uses OpenCV's `SIFT_create()` for keypoint detection
- Uses BFMatcher for descriptor matching
- Applies ratio test (typically 0.75) to filter low-quality matches
- Matches are sorted by quality (inverse distance)
- Supports multiple detectors such as SURF, ORB, etc.

### LOFTR Matching

LOFTR (Local Feature Transformer) is a deep learning-based feature matching method based on Transformer, directly predicting matches from image pairs.

#### Key Functions
1. **Neural Network Matching**: `loftr_compute_matches()` uses a pre-trained Transformer model to directly predict matches between images.
2. **Multi-channel Support**: The `LOFTR` class supports multi-channel image matching.

#### Implementation Details
- Based on Kornia's `LoFTR` implementation
- Supports indoor/outdoor scene models (`method` parameter)
- Input is automatically normalized to [0, 1]
- Matches include confidence scores
- Supports confidence threshold or fixed number of matches
- `LOFTR2` supports image block (quadrant) matching

## Integration with Registration Workflow

Feature matching is a key step in the image alignment workflow, providing the basis for affine and nonlinear registration.

### Homography Estimation

After matches are determined, SpaceMap uses RANSAC to estimate the homography transformation for image alignment:
- Main function: `createHFromPoints2()`
- Input: matched points
- Uses OpenCV's `findHomography()` + RANSAC to compute the transformation
- Applies the transformation and checks alignment quality
- Fine-tunes to minimize alignment error

## Match Visualization

SpaceMap provides match visualization tools for quality control of alignment.

### MatchShow Class

The `MatchShow` class inherits from `AffineBlock` and is used for feature match visualization:
- Displays original points in the first image
- Displays transformed points in the second image
- Connects matched points with lines

## Application in AutoFlow Workflow

Feature matching is mainly used in the AutoFlow automatic registration workflow:
1. `FlowImport` loads data and creates slices
2. Feature matching is used to find correspondences between adjacent slices
3. These correspondences guide affine registration for coarse alignment
4. LDDMM further refines nonlinear transformations

## Best Practices

| Task                    | Recommended Method         | Main Parameters                  |
|-------------------------|---------------------------|-----------------------------------|
| High-contrast tissue    | SIFT                      | `matchr=0.75`, method='sift'      |
| Low-contrast images     | LOFTR                     | model="indoor"                  |
| Multi-channel data      | LOFTR (multiChannel)      | select strongest channel or mean  |
| Large images            | LOFTR2 (block matching)   | quadrant partition                |
| Auto threshold          | `autoSetMatchr()`         | minCount parameter                |

## Key Parameters

| Parameter | Description                              | Typical Value                    |
|-----------|------------------------------------------|----------------------------------|
| matchr    | SIFT ratio threshold/LOFTR confidence    | 0.75 (SIFT), 0.8-0.95 (LOFTR)    |
| method    | Feature detection method                 | 'sift', 'orb', 'surf'            |
| scale     | Whether to normalize images              | True (normalize)                 |
| device    | GPU device for LOFTR                     | CUDA device number               |

## Conclusion

SpaceMap's feature matching system combines traditional computer vision (SIFT) and deep learning (LOFTR) methods, providing flexible solutions for different data types and registration challenges.

---

> For reference source code and detailed implementation, please see the related source files above. 