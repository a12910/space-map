# Installation Guide

This guide provides step-by-step instructions for installing Space-map on your system.

## Prerequisites

Before installing Space-map, ensure you have:

- **Python 3.7 or higher** - Check with `python --version`
- **pip** - Python package installer
- **Git** - For cloning the repository
- **8GB+ RAM** - Recommended for processing large datasets
- **Optional: NVIDIA GPU** - For GPU-accelerated LDDMM

## Installation Methods

### Method 1: From Source (Recommended)

This is the recommended installation method as it gives you access to the latest code and examples.

#### Step 1: Clone the Repository

```bash
git clone https://github.com/a12910/space-map.git
cd space-map
```

#### Step 2: Create Virtual Environment

Using a virtual environment is highly recommended to avoid dependency conflicts.

**On Linux/macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` prefix in your terminal prompt.

#### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirement.txt
```

This will install all required packages including PyTorch, OpenCV, Kornia, and others.

#### Step 4: Install Space-map

**For Development (Recommended):**
```bash
pip install -e .
```

This installs Space-map in "editable" mode, allowing you to modify the code and see changes immediately.

**For Regular Use:**
```bash
pip install .
```

#### Step 5: Verify Installation

```bash
python -c "import space_map; print(f'Space-map version: {space_map.__version__}')"
```

If you see the version number (e.g., "Space-map version: 0.1.0"), installation was successful!

### Method 2: Direct GitHub Installation

For quick installation without cloning:

```bash
# Create and activate virtual environment first
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# venv\Scripts\activate   # On Windows

# Install directly from GitHub
pip install git+https://github.com/a12910/space-map.git
```

## GPU Support (Optional)

For GPU-accelerated LDDMM registration, you need CUDA-enabled PyTorch.

### Check GPU Availability

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
```

### Install CUDA-enabled PyTorch

If you have an NVIDIA GPU, install the appropriate PyTorch version:

**For CUDA 11.8:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**For CPU-only:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) for the latest installation commands.

## Platform-Specific Notes

### Linux

Most dependencies should install smoothly. If you encounter issues, install build tools:

```bash
sudo apt-get update
sudo apt-get install python3-dev build-essential
```

### macOS

Install Xcode command line tools if needed:

```bash
xcode-select --install
```

For Apple Silicon (M1/M2/M3), PyTorch will use MPS acceleration automatically.

### Windows

- Install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) for some dependencies
- Use PowerShell or Command Prompt (not Git Bash) for virtual environment activation

## Troubleshooting

### Common Issues

**Problem: "ModuleNotFoundError: No module named 'space_map'"**

Solution: Make sure you're in the virtual environment and installed the package:
```bash
which python  # Should show venv/bin/python
pip install -e .
```

**Problem: OpenCV import errors**

Solution: Try the headless version:
```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

**Problem: PyTorch CUDA not working**

Solution: Reinstall PyTorch with correct CUDA version:
```bash
pip uninstall torch torchvision
# Then install with appropriate CUDA version (see GPU Support section)
```

**Problem: Out of memory during installation**

Solution: Install packages one at a time:
```bash
pip install numpy pandas
pip install torch
pip install opencv-python
pip install -r requirement.txt
```

## Verifying Your Installation

Run this complete test:

```python
import space_map
import torch
import cv2
import pandas as pd
import numpy as np

print(f"Space-map version: {space_map.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"OpenCV version: {cv2.__version__}")
print("✓ All core dependencies loaded successfully!")
```

## Updating Space-map

If you installed in development mode (`pip install -e .`), update with:

```bash
cd space-map
git pull origin master
pip install -r requirement.txt  # Update dependencies if needed
```

## Uninstallation

To remove Space-map:

```bash
pip uninstall space-map
```

To remove the virtual environment:

```bash
deactivate  # Exit virtual environment
rm -rf venv  # Delete virtual environment directory
```

## Dependencies Overview

Space-map requires the following packages:

### Core Scientific
- **numpy** - Numerical computing
- **pandas** - Data manipulation
- **scipy** - Scientific algorithms
- **scikit-learn** - Machine learning utilities

### Computer Vision
- **opencv-python** - Image processing
- **scikit-image** - Image algorithms
- **tifffile** - TIFF image I/O
- **nibabel** - Medical imaging formats

### Deep Learning
- **torch** - PyTorch deep learning framework
- **kornia** - Differentiable computer vision

### Visualization
- **matplotlib** - Plotting
- **seaborn** - Statistical visualization

### Performance
- **numba** - JIT compilation for speed
- **tqdm** - Progress bars

### Registration
- **pycpd** - Coherent Point Drift algorithm

See [requirement.txt](https://github.com/a12910/space-map/blob/master/requirement.txt) for the complete list with version requirements.

## Next Steps

After successful installation:

1. **[Quick Start Guide](quickstart.md)** - Learn the basic workflow
2. **[Example Notebooks](https://github.com/a12910/space-map/tree/master/examples)** - Try interactive tutorials
3. **[GitHub Repository](https://github.com/a12910/space-map)** - Explore the source code

## Getting Help

If you encounter issues not covered here:

- Check [GitHub Issues](https://github.com/a12910/space-map/issues) for similar problems
- Create a new issue with:
  - Your error message
  - Python version (`python --version`)
  - Operating system
  - Installation method used
- Join [GitHub Discussions](https://github.com/a12910/space-map/discussions) for community support

---

**Installation complete?** Continue to the [Quick Start Guide](quickstart.md) to begin using Space-map!
