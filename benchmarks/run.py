#!/usr/bin/env python
"""
Space-map alignment script — one command to align your spatial data.

Usage:
    python benchmarks/run.py examples/cells2.csv.gz
    python benchmarks/run.py your_data.csv --output results/
    python benchmarks/run.py your_data.csv --layer-col section --x-col cx --y-col cy

Input:
    A CSV (or .csv.gz) file with columns for x, y coordinates and a layer/section ID.

Output:
    - aligned.csv.gz  : aligned coordinates
    - metrics.csv     : per-layer-pair alignment quality (Dice, SSIM, PSNR, MSE)
    - summary.json    : mean metrics across all layer pairs
"""

import matplotlib
matplotlib.use("Agg")

import argparse
import json
import os
import sys
import time

# Ensure the project root is on sys.path so `import space_map` works
# even when running from the repo without installing.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd


def load_data(path, x_col, y_col, layer_col):
    """Load CSV and split into per-layer coordinate arrays."""
    print(f"Loading {path} ...")
    df = pd.read_csv(path)

    for col in [x_col, y_col, layer_col]:
        if col not in df.columns:
            sys.exit(f"Error: column '{col}' not found. Available: {list(df.columns)}")

    layers = sorted(df[layer_col].unique())
    xys = []
    ids = []
    for layer in layers:
        sub = df[df[layer_col] == layer]
        xys.append(sub[[x_col, y_col]].values.astype(np.float64))
        ids.append(layer)

    total = sum(len(x) for x in xys)
    print(f"  {len(layers)} layers, {total:,} cells total")
    return xys, ids, df


def run_spacemap(xys, ids):
    """Run Space-map two-stage registration (affine + LDDMM)."""
    import space_map
    from space_map import Slice
    from space_map.flow import FlowImport, AutoFlowMultiCenter4
    import tempfile

    base_path = tempfile.mkdtemp(prefix="spacemap_bench_")

    # Init project
    flow = FlowImport(base_path)
    ids_str = [str(i) for i in ids]
    slices = flow.init_xys(xys, ids_str)

    # Stage 1: Affine alignment
    print("  Stage 1/2: Affine registration ...")
    af = AutoFlowMultiCenter4(slices, initJKey=Slice.rawKey)
    af.affine(useKey="DF", show=False)

    # Stage 2: LDDMM non-rigid alignment
    print("  Stage 2/2: LDDMM registration ...")
    af.ldm_pair(fromKey=Slice.align1Key, toKey=Slice.align2Key, show=False)

    # Extract results
    aligned = [s.ps(Slice.align2Key) for s in slices]
    return aligned


def evaluate(raw_xys, aligned_xys, ids):
    """Compute alignment quality metrics for each adjacent layer pair."""
    import space_map
    from space_map.find.err import err_dice, err_ssim, err_psnr, err_mse

    metrics = {"dice": err_dice, "ssim": err_ssim, "psnr": err_psnr, "mse": err_mse}
    records = []
    for i in range(len(ids) - 1):
        raw_i = space_map.show_img(raw_xys[i])
        raw_j = space_map.show_img(raw_xys[i + 1])
        ali_i = space_map.show_img(aligned_xys[i])
        ali_j = space_map.show_img(aligned_xys[i + 1])

        row = {"layer_i": ids[i], "layer_j": ids[i + 1]}
        for name, func in metrics.items():
            row[f"{name}_raw"] = float(func(raw_i, raw_j))
            row[f"{name}_aligned"] = float(func(ali_i, ali_j))
        records.append(row)

    return pd.DataFrame(records)


def save_aligned_csv(raw_df, aligned_xys, ids, x_col, y_col, layer_col, output_path):
    """Write aligned coordinates back to CSV, preserving all original columns."""
    df = raw_df.copy()
    layers = sorted(df[layer_col].unique())
    for layer, xy in zip(layers, aligned_xys):
        mask = df[layer_col] == layer
        df.loc[mask, x_col] = xy[:, 0]
        df.loc[mask, y_col] = xy[:, 1]
    df.to_csv(output_path, index=False)
    print(f"  Aligned coordinates -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Align spatial transcriptomics data with Space-map",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmarks/run.py examples/cells2.csv.gz
  python benchmarks/run.py data.csv --output my_results/
  python benchmarks/run.py data.csv --x-col cx --y-col cy --layer-col section
        """,
    )
    parser.add_argument("input", help="Path to CSV file (x, y, layer columns)")
    parser.add_argument("--output", "-o", default="benchmarks/results", help="Output directory (default: benchmarks/results)")
    parser.add_argument("--x-col", default="x", help="Column name for x coordinates (default: x)")
    parser.add_argument("--y-col", default="y", help="Column name for y coordinates (default: y)")
    parser.add_argument("--layer-col", default="layer", help="Column name for layer/section ID (default: layer)")
    args = parser.parse_args()

    # Load
    xys, ids, raw_df = load_data(args.input, args.x_col, args.y_col, args.layer_col)

    # Align
    print("Running Space-map alignment ...")
    t0 = time.time()
    aligned_xys = run_spacemap(xys, ids)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # Evaluate
    print("Computing metrics ...")
    metrics_df = evaluate(xys, aligned_xys, ids)

    # Summary
    summary = {"time_seconds": round(elapsed, 2)}
    for col in metrics_df.columns:
        if col.startswith("layer"):
            continue
        vals = metrics_df[col].dropna()
        summary[f"{col}_mean"] = round(float(vals.mean()), 6)
        summary[f"{col}_std"] = round(float(vals.std()), 6)

    # Save
    os.makedirs(args.output, exist_ok=True)
    save_aligned_csv(raw_df, aligned_xys, ids, args.x_col, args.y_col, args.layer_col,
                     os.path.join(args.output, "aligned.csv.gz"))
    metrics_df.to_csv(os.path.join(args.output, "metrics.csv"), index=False)
    print(f"  Metrics -> {os.path.join(args.output, 'metrics.csv')}")
    with open(os.path.join(args.output, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary -> {os.path.join(args.output, 'summary.json')}")

    # Print summary
    print("\n--- Results ---")
    print(f"  Dice:  {summary['dice_raw_mean']:.4f} -> {summary['dice_aligned_mean']:.4f}")
    print(f"  SSIM:  {summary['ssim_raw_mean']:.4f} -> {summary['ssim_aligned_mean']:.4f}")
    print(f"  PSNR:  {summary['psnr_raw_mean']:.2f} -> {summary['psnr_aligned_mean']:.2f}")
    print(f"  MSE:   {summary['mse_raw_mean']:.6f} -> {summary['mse_aligned_mean']:.6f}")


if __name__ == "__main__":
    main()
