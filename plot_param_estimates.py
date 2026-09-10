#!/usr/bin/env python
"""Plot scalar TOMBOMBADIL MAP parameter estimates from saved CSV files.

Reads CSV files written by ``tombombadil.sample.save_params``:

    scalar_STEM_Allparams.csv  with columns: variable, value

In stem mode, produces ``STEM_scalar_plot.png`` by default. In folder mode,
    writes one PNG for each ``scalar_*_Allparams.csv`` file found in the input folder.
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np


_PREFERRED_SCALAR_ORDER = [
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "theta",
    "omega",
]


def _require_columns(path, names, required):
    missing = [name for name in required if name not in names]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"{path} is missing required column(s): {missing_str}")


def load_scalar_csv(path):
    """Load scalar parameter estimates from a CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scalar CSV not found: {path}")

    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    if data.dtype.names is None:
        raise ValueError(f"{path} does not look like a named-column CSV")

    _require_columns(path, data.dtype.names, ["variable", "value"])
    data = np.atleast_1d(data)
    return {str(row["variable"]): float(row["value"]) for row in data}


def find_scalar_csvs(folder):
    """Find scalar-mode parameter CSV files in a folder."""
    paths = sorted(glob.glob(os.path.join(folder, "scalar_*_Allparams.csv")))
    if not paths:
        raise FileNotFoundError(f"No scalar_*_Allparams.csv files found in {folder}")
    return paths


def _ordered_scalar_names(scalar_params):
    preferred = [p for p in _PREFERRED_SCALAR_ORDER if p in scalar_params]
    extras = sorted(p for p in scalar_params if p not in preferred)
    return preferred + extras


def plot_scalar_params(scalar_params, log_scale=False):
    """Create a scalar parameter bar plot."""
    names = _ordered_scalar_names(scalar_params)
    values = np.array([scalar_params[name] for name in names], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(len(names)), values, color="steelblue", alpha=0.85)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel("Estimate")
    ax.set_title("Scalar parameter estimates")
    if log_scale:
        ax.set_yscale("log")
    plt.tight_layout()
    return fig


def plot_one(scalar_path, scalar_out, log_scale=False):
    scalar_params = load_scalar_csv(scalar_path)
    fig = plot_scalar_params(scalar_params, log_scale=log_scale)
    fig.savefig(scalar_out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote scalar plot: {scalar_out}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot TOMBOMBADIL scalar parameter estimates from CSV files."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--stem",
        help="Input stem. Reads scalar_STEM_Allparams.csv.",
    )
    input_group.add_argument(
        "--folder",
        help="Input folder. Plots every scalar_*_Allparams.csv file in the folder.",
    )
    parser.add_argument(
        "--output",
        help="Output PNG path in stem mode, or output folder in folder mode.",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Plot estimates on a log y-axis.",
    )
    args = parser.parse_args()

    if args.folder:
        output_folder = args.output or args.folder
        os.makedirs(output_folder, exist_ok=True)
        for scalar_path in find_scalar_csvs(args.folder):
            stem_name = os.path.basename(scalar_path[: -len("_Allparams.csv")])
            scalar_out = os.path.join(output_folder, stem_name + "_plot.png")
            plot_one(scalar_path, scalar_out, log_scale=args.log_scale)
    else:
        scalar_path = os.path.join(
            os.path.dirname(args.stem),
            "scalar_" + os.path.basename(args.stem) + "_Allparams.csv",
        )
        scalar_out = args.output or os.path.join(
            os.path.dirname(args.stem),
            "scalar_" + os.path.basename(args.stem) + "_plot.png",
        )
        plot_one(scalar_path, scalar_out, log_scale=args.log_scale)


if __name__ == "__main__":
    main()
