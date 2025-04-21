# Changelog

All notable changes to SpaceMap will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial documentation website
- Improved API documentation

## [0.1.0] - YYYY-MM-DD

### Added
- Initial release of SpaceMap
- Basic registration functionality
  - Affine registration for coarse alignment
  - Flow-based fine registration with LDDMM
- Feature matching using SIFT and LoFTR
- Utils for visualization and data processing
- Support for multi-modal data integration
- GPU acceleration for transform computations

### Known Issues
- Large datasets (>10M points) might require significant memory resources
- Some visualization functions may be slow with very large datasets 