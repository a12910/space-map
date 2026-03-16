# Space-map Reproducible Examples

Align your spatial transcriptomics serial sections with one command or one notebook.

## Quick Start

### Option 1: Command line

```bash
# Install
pip install git+https://github.com/a12910/space-map.git

# Quick test with toy data (~1 minute)
python benchmarks/run.py examples/toy_data.csv.gz

# Full run with example data (~1 hour)
python benchmarks/run.py examples/cells2.csv.gz

# Run with your own data
python benchmarks/run.py your_data.csv --output results/
```

### Option 2: Jupyter notebook

```bash
pip install space-map jupyter
jupyter notebook benchmarks/example_notebook.ipynb
```

Open the notebook, change the `DATA_PATH` to your file, and run all cells.

## Input Format

A CSV file (`.csv` or `.csv.gz`) with these columns:

| Column | Description | Example |
|--------|-------------|---------|
| `x` | x coordinate of each cell | 844.05 |
| `y` | y coordinate of each cell | 2610.31 |
| `layer` | Section / layer ID | 1 |

If your columns have different names, specify them:

```bash
python benchmarks/run.py data.csv --x-col cx --y-col cy --layer-col section
```

## Output

The script produces three files in the output directory:

| File | Description |
|------|-------------|
| `aligned.csv.gz` | Your data with aligned x, y coordinates |
| `metrics.csv` | Per-layer-pair quality metrics |
| `summary.json` | Mean metrics across all pairs |

## Quality Metrics

Each adjacent pair of layers is compared before and after alignment:

| Metric | What it measures | Better |
|--------|-----------------|--------|
| **Dice** | Overlap of cell density | Higher (max 1.0) |
| **SSIM** | Structural similarity | Higher (max 1.0) |
| **PSNR** | Signal-to-noise ratio | Higher |
| **MSE** | Mean squared error | Lower (min 0.0) |

## File Structure

```
benchmarks/
├── run.py                 # Command-line script
├── example_notebook.ipynb # Step-by-step Jupyter notebook
├── evaluate.py            # Evaluation utilities
└── results/               # Output directory
```
