"""Evaluation utilities for Space-map alignment quality.

Computes per-layer-pair metrics (Dice, SSIM, PSNR, MSE) by rendering
coordinates to density images and comparing adjacent layers.
"""

import os

import numpy as np
import pandas as pd


def evaluate(raw_xys, aligned_xys, ids, metrics=None):
    """Compare adjacent layers before/after alignment.

    Parameters
    ----------
    raw_xys : list of np.ndarray
        Original per-layer coordinates.
    aligned_xys : list of np.ndarray
        Aligned per-layer coordinates.
    ids : list
        Layer identifiers.
    metrics : list of str, optional
        Which metrics to compute. Default: all four.

    Returns
    -------
    pd.DataFrame
        Per-layer-pair scores.
    """
    import space_map
    from space_map.find.err import err_dice, err_ssim, err_psnr, err_mse

    all_metrics = {"dice": err_dice, "ssim": err_ssim, "psnr": err_psnr, "mse": err_mse}

    if metrics is not None:
        all_metrics = {k: v for k, v in all_metrics.items() if k in metrics}

    records = []
    for i in range(len(ids) - 1):
        raw_i = space_map.show_img(raw_xys[i])
        raw_j = space_map.show_img(raw_xys[i + 1])
        ali_i = space_map.show_img(aligned_xys[i])
        ali_j = space_map.show_img(aligned_xys[i + 1])

        row = {"layer_i": ids[i], "layer_j": ids[i + 1]}
        for name, func in all_metrics.items():
            row[f"{name}_raw"] = float(func(raw_i, raw_j))
            row[f"{name}_aligned"] = float(func(ali_i, ali_j))
        records.append(row)

    return pd.DataFrame(records)


def summarise(metrics_df):
    """Compute mean/std summary from per-layer metrics DataFrame."""
    summary = {}
    for col in metrics_df.columns:
        if col.startswith("layer"):
            continue
        vals = metrics_df[col].dropna()
        summary[f"{col}_mean"] = round(float(vals.mean()), 6)
        summary[f"{col}_std"] = round(float(vals.std()), 6)
    return summary
