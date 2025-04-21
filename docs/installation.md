# Installation Guide

There are several ways to install SpaceMap. Choose the method that best fits your needs.

## Installing via pip (Recommended)

The simplest way to install SpaceMap is via pip:

```bash
pip install spacemap
```

This will install SpaceMap and all its dependencies automatically.

## Installing from Source

For the latest features or development purposes, you can install SpaceMap directly from the GitHub repository:

```bash
git clone https://github.com/a12910/spacemap.git
cd spacemap
pip install -e .
```

The `-e` flag installs the package in "editable" mode, which means changes to the source code will be reflected in your environment without reinstallation.

## Dependencies

SpaceMap requires the following Python packages:

```
opencv-python
pandas
numpy
torch
kornia
scikit-learn
tifffile
matplotlib
numba
scipy
tqdm
nibabel
seaborn
scikit-image
```

These dependencies will be automatically installed when installing via pip. If you're installing manually, you can install all required dependencies with:

```bash
pip install -r requirement.txt
```

## Optional Dependencies

For optimal performance, we recommend the following:

### GPU Support

SpaceMap can leverage GPU acceleration for faster processing:

1. Make sure you have a CUDA-compatible NVIDIA GPU
2. Install the appropriate CUDA toolkit version compatible with your PyTorch version
3. Install the GPU version of PyTorch:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Replace `cu118` with your CUDA version as needed.

## Verifying Installation

You can verify your installation by importing SpaceMap in Python:

```python
import spacemap as sm
print(sm.__version__)
```

If the installation was successful, this will print the version number without any errors.

## Troubleshooting

If you encounter issues during installation:

1. Ensure your Python version is 3.7 or higher
2. Update pip: `pip install --upgrade pip`
3. Check for conflicts with other installed packages
4. Try installing in a fresh virtual environment:

```bash
python -m venv spacemap_env
source spacemap_env/bin/activate  # On Windows: spacemap_env\Scripts\activate
pip install spacemap
```

For more help, please [open an issue](https://github.com/yourusername/spacemap/issues) on GitHub. 