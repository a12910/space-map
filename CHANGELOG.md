# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Add comprehensive test suite
- Add more example notebooks
- Performance optimizations for large datasets
- Support for additional spatial transcriptomics platforms

## [0.1.0] - 2024-01-05

### Added
- Initial release of Space-map
- Multi-modal registration combining cell coordinates, types, and histological images
- Two-stage registration: affine transformation followed by LDDMM
- Feature matching using LoFTR and SIFT
- GPU-accelerated LDDMM implementation
- Support for Xenium and CODEX data formats
- FlowImport/FlowExport pipeline for data management
- Comprehensive documentation with MkDocs
- Example notebooks for quick start and advanced usage

### Features
- `flow.FlowImport`: Data import and initialization
- `flow.AutoFlowMultiCenter4/5`: Automated registration workflows
- `base.Slice`: Tissue section management
- `registration.lddmm`: Deformable registration
- `matches`: Feature matching algorithms (SIFT, LoFTR)
- `flow.FlowExport`: Results export and visualization

### Documentation
- Complete API reference
- Quick start guide
- Installation instructions
- Architecture overview
- Example workflows

### Infrastructure
- GitHub Actions for documentation deployment
- GitHub Actions for PyPI publishing
- MkDocs with Material theme
- PyPI package distribution

## Release Notes

### [0.1.0] - Initial Release

This is the first public release of Space-map, providing a complete framework for reconstructing atlas-level single-cell 3D tissue maps from serial sections.

**Key Highlights:**
- Handles large-scale datasets (2-3M cells)
- ~2-fold more accurate than existing methods (PASTE, STalign)
- Runs on standard laptop hardware
- Comprehensive documentation and examples

**Tested Platforms:**
- Xenium spatial transcriptomics
- CODEX spatial proteomics

**System Requirements:**
- Python 3.7+
- PyTorch for GPU acceleration
- 8GB+ RAM recommended

---

[Unreleased]: https://github.com/a12910/space-map/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/a12910/space-map/releases/tag/v0.1.0
