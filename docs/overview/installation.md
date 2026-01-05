# Installation Guide

This guide will help you install Space-map on your system.

## Prerequisites

- Python 3.7 or higher
- pip package manager
- Git (for installation from source)

## Recommended Installation (From Source)

This is the current method to install Space-map until it's published on PyPI.

### Step 1: Clone the Repository

```bash
git clone https://github.com/a12910/space-map.git
cd space-map
```

### Step 2: Create Virtual Environment

It's highly recommended to use a virtual environment:

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

### Step 3: Install Dependencies

```bash
pip install -r requirement.txt
```

### Step 4: Install Space-map

Install in development mode (recommended for updates):
```bash
pip install -e .
```

Or install normally:
```bash
pip install .
```

### Step 5: Verify Installation

```bash
python -c "import spacemap; print(spacemap.__version__)"
```

You should see the version number printed.

## Alternative: Direct Installation from GitHub

If you don't need to modify the code:

```bash
# Create and activate virtual environment first
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# venv\Scripts\activate   # On Windows

# Install directly from GitHub
pip install git+https://github.com/a12910/space-map.git
```

## Dependencies

Space-map requires the following main packages:

- **Core**: numpy, pandas, scipy
- **Computer Vision**: opencv-python, scikit-image
- **Deep Learning**: torch, kornia
- **Scientific**: scikit-learn, numba
- **Visualization**: matplotlib, seaborn
- **Image Processing**: tifffile, nibabel

See [`requirement.txt`](../../requirement.txt) for the complete list.

## GPU Support (Optional but Recommended)

For GPU acceleration with LDDMM:

### CUDA-enabled PyTorch

If you have an NVIDIA GPU:

```bash
# Install PyTorch with CUDA support
# Visit https://pytorch.org/get-started/locally/ for the right command
# Example for CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Space-map will automatically use GPU if available.

## Platform-Specific Notes

### Linux

Should work out of the box. Ensure you have development tools:

```bash
sudo apt-get update
sudo apt-get install python3-dev build-essential
```

### macOS

```bash
# Install Xcode command line tools if needed
xcode-select --install
```

### Windows

- Ensure Visual C++ Build Tools are installed for some dependencies
- Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

## Troubleshooting

### Import Errors

If you get import errors:

```bash
# Make sure you're in the virtual environment
which python  # Should point to venv/bin/python

# Reinstall dependencies
pip install -r requirement.txt
```

### OpenCV Issues

If OpenCV doesn't work:

```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

### PyTorch CUDA Issues

To check if PyTorch can use GPU:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

## Updating Space-map

If you installed in development mode (`pip install -e .`):

```bash
cd space-map
git pull origin master
pip install -r requirement.txt  # Update dependencies if needed
```

## Uninstallation

```bash
pip uninstall spacemap
```

## Coming Soon: PyPI Installation

Space-map will be available on PyPI soon, allowing simple installation:

```bash
# Future (not available yet)
pip install spacemap
```

## Next Steps

After installation:
- Read the [Quick Start Guide](quickstart.md)
- Try the [example notebooks](../../examples/)
- Explore the [documentation](https://a12910.github.io/space-map)

## Getting Help

If you encounter installation issues:
- Check the [GitHub Issues](https://github.com/a12910/space-map/issues)
- Create a new issue with your error message and system info 