#!/usr/bin/env python3
"""
Standalone analysis script for Surya predictions.

Usage:
    python analyze_predictions.py --prediction-nc path/to/prediction.nc
    python analyze_predictions.py --prediction-nc path/to/prediction.nc --channel aia171
    python analyze_predictions.py --prediction-nc path/to/prediction.nc --all-channels
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

CHANNELS = [
    "aia94",
    "aia131",
    "aia171",
    "aia193",
    "aia211",
    "aia304",
    "aia335",
    "aia1600",
    "hmi_m",
    "hmi_bx",
    "hmi_by",
    "hmi_bz",
    "hmi_v",
]


def compute_mse(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute MSE on valid (finite) pixels."""
    valid = np.isfinite(pred) & np.isfinite(gt)
    if not valid.any():
        return float("nan")
    return float(np.mean((pred[valid] - gt[valid]) ** 2))


def analyze_channel(
    ds: xr.Dataset, channel: str, output_dir: Path, show_plots: bool = True
) -> list[float] | None:
    """Analyze a single channel."""
    print(f"\nAnalyzing channel: {channel}")
    print("-" * 40)

    if channel not in ds:
        print(f"Channel {channel} not found in dataset")
        return None

    gt_channel = f"gt_{channel}"
    if gt_channel not in ds:
        print(f"Ground truth {gt_channel} not found")
        return None

    pred = ds[channel].values[0]  # [prediction_time, y, x]
    gt = ds[gt_channel].values[0]

    n_steps = pred.shape[0]
    mse_per_step = []

    print("\nMSE per prediction step:")
    print(f"{'Step':<6} {'MSE':<15}")
    print("-" * 21)

    for step in range(n_steps):
        mse = compute_mse(pred[step], gt[step])
        mse_per_step.append(mse)
        print(f"{step + 1:<6} {mse:<15.6f}")

    avg_mse = np.nanmean(mse_per_step)
    print(f"\nAverage MSE: {avg_mse:.6f}")

    return mse_per_step


def plot_mse_trend(mse_dict: dict[str, list[float]], output_dir: Path):
    """Plot MSE trend for all channels."""
    plt.figure(figsize=(10, 6))

    for channel, mse_values in mse_dict.items():
        steps = range(1, len(mse_values) + 1)
        plt.plot(steps, mse_values, marker="o", label=channel)

    plt.xlabel("Prediction Step")
    plt.ylabel("MSE")
    plt.title("MSE vs Prediction Step")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = output_dir / "mse_trend.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved: {plot_path}")
    plt.close()


def plot_comparison(ds: xr.Dataset, channel: str, output_dir: Path, n_steps: int = 3):
    """Plot GT vs Prediction comparison."""
    gt_channel = f"gt_{channel}"

    if channel not in ds or gt_channel not in ds:
        print(f"Cannot plot comparison for {channel}: missing data")
        return

    pred = ds[channel].values[0]
    gt = ds[gt_channel].values[0]

    n_plot = min(n_steps, pred.shape[0])

    fig, axes = plt.subplots(n_plot, 3, figsize=(12, 4 * n_plot))
    if n_plot == 1:
        axes = axes.reshape(1, -1)

    for step in range(n_plot):
        pred_step = np.log1p(np.abs(pred[step]))
        gt_step = np.log1p(np.abs(gt[step]))
        diff = pred[step] - gt[step]

        axes[step, 0].imshow(gt_step, origin="lower", cmap="viridis")
        axes[step, 0].set_title(f"Step {step + 1}: Ground Truth")
        axes[step, 0].axis("off")

        axes[step, 1].imshow(pred_step, origin="lower", cmap="viridis")
        axes[step, 1].set_title(f"Step {step + 1}: Prediction")
        axes[step, 1].axis("off")

        diff_std = np.nanstd(diff)
        if diff_std > 0:
            vmin, vmax = -diff_std * 3, diff_std * 3
        else:
            vmin, vmax = -1, 1

        im = axes[step, 2].imshow(diff, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[step, 2].set_title(f"Step {step + 1}: Difference")
        axes[step, 2].axis("off")
        plt.colorbar(im, ax=axes[step, 2], fraction=0.046)

    plt.suptitle(f"Channel: {channel}", fontsize=14)
    plt.tight_layout()

    plot_path = output_dir / f"comparison_{channel}.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved: {plot_path}")
    plt.close()


def rank_channels(mse_dict: dict[str, float]):
    """Rank channels by average MSE."""
    print("\nCHANNEL RANKING (by average MSE)")
    print("=" * 40)
    print(f"{'Rank':<6} {'Channel':<12} {'Avg MSE':<15}")
    print("-" * 33)

    sorted_channels = sorted(mse_dict.items(), key=lambda x: x[1])
    for rank, (channel, mse) in enumerate(sorted_channels, 1):
        print(f"{rank:<6} {channel:<12} {mse:<15.6f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Surya predictions")
    parser.add_argument("--prediction-nc", required=True, help="Path to prediction.nc")
    parser.add_argument("--channel", default="aia94", help="Channel to analyze")
    parser.add_argument("--all-channels", action="store_true", help="Analyze all channels")
    parser.add_argument("--show-mse-trend", action="store_true", help="Plot MSE trend")
    parser.add_argument("--rank-channels", action="store_true", help="Rank channels by error")
    parser.add_argument("--n-plot-steps", type=int, default=3, help="Number of steps to plot")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")

    args = parser.parse_args()

    nc_path = Path(args.prediction_nc)
    if not nc_path.exists():
        print(f"Error: {nc_path} not found")
        return 1

    output_dir = nc_path.parent
    print(f"\nLoading: {nc_path}")

    ds = xr.open_dataset(nc_path)

    # List available variables
    pred_vars = [v for v in ds.data_vars if not v.startswith("gt_") and v in CHANNELS]
    gt_vars = [v for v in ds.data_vars if v.startswith("gt_")]
    print(f"Prediction variables: {len(pred_vars)}")
    print(f"Ground truth variables: {len(gt_vars)}")

    # Determine channels to analyze
    if args.all_channels:
        channels = [ch for ch in CHANNELS if ch in ds]
    else:
        channels = [args.channel]

    if not channels:
        print("No valid channels found in dataset")
        ds.close()
        return 1

    mse_results: dict[str, list[float]] = {}
    avg_mse_results: dict[str, float] = {}

    for channel in channels:
        mse_per_step = analyze_channel(ds, channel, output_dir)
        if mse_per_step:
            mse_results[channel] = mse_per_step
            avg_mse_results[channel] = float(np.nanmean(mse_per_step))
            if not args.no_plots:
                plot_comparison(ds, channel, output_dir, args.n_plot_steps)

    if args.show_mse_trend and mse_results and not args.no_plots:
        plot_mse_trend(mse_results, output_dir)

    if args.rank_channels and avg_mse_results:
        rank_channels(avg_mse_results)

    ds.close()
    print("\nAnalysis complete!")
    return 0


if __name__ == "__main__":
    exit(main())
