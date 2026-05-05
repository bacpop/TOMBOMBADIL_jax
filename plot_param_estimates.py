#!/usr/bin/env python
"""Plot TOMBOMBADIL MAP parameter estimates from saved CSV files.

Reads CSV pairs written by ``tombombadil.sample.save_params``:

    STEM_omega.csv   with columns: site, omega_map, variant
    STEM_scalar.csv  with columns: variable, value

In stem mode, produces two PNG files by default:

    STEM_omega_plot.png
    STEM_scalar_plot.png

In folder mode, writes one pair of PNG files per complete CSV pair found in
the input folder.
"""

import argparse
import csv
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
    "eta",
    "theta",
]


def _require_columns(path, names, required):
    missing = [name for name in required if name not in names]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"{path} is missing required column(s): {missing_str}")


def load_omega_csv(path):
    """Load per-site omega estimates from an omega CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Omega CSV not found: {path}")

    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.dtype.names is None:
        raise ValueError(f"{path} does not look like a named-column CSV")

    _require_columns(path, data.dtype.names, ["site", "omega_map"])

    sites = np.atleast_1d(data["site"]).astype(int)
    omega = np.atleast_1d(data["omega_map"]).astype(np.float64)
    variant = None
    if "variant" in data.dtype.names:
        variant = np.atleast_1d(data["variant"]).astype(int)
        if len(variant) != len(omega):
            raise ValueError(f"{path} has mismatched omega_map and variant lengths")

    if len(sites) != len(omega):
        raise ValueError(f"{path} has mismatched site and omega_map lengths")

    return sites, omega, variant


def load_scalar_csv(path, include_eta=True):
    """Load scalar parameters from a scalar CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scalar CSV not found: {path}")

    scalars = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} does not look like a named-column CSV")
        _require_columns(path, reader.fieldnames, ["variable", "value"])
        for row in reader:
            variable = row["variable"]
            if variable == "":
                continue
            scalars[variable] = float(row["value"])

    if include_eta and "eta" not in scalars:
        scalars["eta"] = 1.0

    return scalars


def find_csv_pairs(folder):
    """Find complete *_omega.csv / *_scalar.csv pairs in a folder."""
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"Folder not found: {folder}")

    pairs = []
    omega_paths = sorted(glob.glob(os.path.join(folder, "*_omega.csv")))
    for omega_path in omega_paths:
        stem = omega_path[: -len("_omega.csv")]
        scalar_path = stem + "_scalar.csv"
        if not os.path.exists(scalar_path):
            print(f"Skipping {omega_path}: matching scalar CSV not found")
            continue
        pairs.append((stem, omega_path, scalar_path))

    if not pairs:
        raise FileNotFoundError(f"No complete *_omega.csv / *_scalar.csv pairs found in {folder}")

    return pairs


def ordered_scalar_names(scalars):
    """Return scalar names in the standard order, followed by extras."""
    preferred = [name for name in _PREFERRED_SCALAR_ORDER if name in scalars]
    extras = sorted(name for name in scalars if name not in _PREFERRED_SCALAR_ORDER)
    return preferred + extras


def plot_omega(sites, omega, variant=None, log_scale=True):
    """Create a per-site omega scatter plot."""
    fig, ax = plt.subplots(figsize=(16, 4))

    if variant is None:
        ax.scatter(
            sites,
            omega,
            color="black",
            s=8,
            alpha=0.65,
            linewidths=0,
            label="Omega estimate",
            zorder=3,
        )
    else:
        invariant = variant == 0
        variable = variant != 0
        if invariant.any():
            ax.scatter(
                sites[invariant],
                omega[invariant],
                color="lightgrey",
                s=7,
                alpha=0.7,
                linewidths=0,
                label="Invariant site",
                zorder=2,
            )
        if variable.any():
            ax.scatter(
                sites[variable],
                omega[variable],
                color="black",
                s=9,
                alpha=0.75,
                linewidths=0,
                label="Variable site",
                zorder=3,
            )

    ax.axhline(
        1.0,
        color="red",
        linestyle="--",
        linewidth=1.1,
        alpha=0.8,
        label="omega = 1",
        zorder=4,
    )
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("omega (dN/dS, log scale)")
    else:
        ax.set_ylabel("omega (dN/dS)")
    ax.set_xlabel("Alignment site")
    ax.set_title("Per-site omega estimates")
    ax.set_xlim(sites.min() - 1, sites.max() + 1)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=8)
    plt.tight_layout()
    return fig


def plot_scalars(scalars):
    """Create a scalar-parameter dot plot."""
    names = ordered_scalar_names(scalars)
    values = np.array([scalars[name] for name in names], dtype=np.float64)
    y = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(7.5, len(names) * 0.55 + 1.5))
    ax.scatter(
        values,
        y,
        color="black",
        s=50,
        marker="D",
        edgecolors="black",
        linewidths=0.4,
        zorder=3,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    if np.all(values > 0):
        ax.set_xscale("log")
        ax.set_xlabel("Parameter value (log scale)")
    else:
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_xlabel("Parameter value (signed log scale)")
    ax.set_title("Scalar parameter estimates")
    ax.axvline(1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.tight_layout()
    return fig


def write_plots(
    omega_path,
    scalar_path,
    omega_out,
    scalar_out,
    log_omega=False,
    include_eta=True,
):
    """Load one CSV pair and write omega/scalar plot PNGs."""
    sites, omega, variant = load_omega_csv(omega_path)
    scalars = load_scalar_csv(scalar_path, include_eta=include_eta)

    omega_fig = plot_omega(sites, omega, variant, log_scale=log_omega)
    omega_fig.savefig(omega_out, dpi=300, bbox_inches="tight")
    plt.close(omega_fig)

    scalar_fig = plot_scalars(scalars)
    scalar_fig.savefig(scalar_out, dpi=300, bbox_inches="tight")
    plt.close(scalar_fig)

    print(f"Wrote omega plot: {omega_out}")
    print(f"Wrote scalar plot: {scalar_out}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot TOMBOMBADIL omega and scalar parameter estimates from CSV files."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--stem",
        help="Input stem. Reads STEM_omega.csv and STEM_scalar.csv.",
    )
    source.add_argument(
        "--folder",
        help="Input folder. Plots every complete *_omega.csv / *_scalar.csv pair in the folder.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help=(
            "Output prefix for --stem mode, or output folder for --folder mode. "
            "Defaults to the input stem or folder."
        ),
    )

    omega_scale = parser.add_mutually_exclusive_group()
    omega_scale.add_argument(
        "--log-omega",
        dest="log_omega",
        action="store_true",
        default=False,
        help="Plot omega on a log y-axis.",
    )
    omega_scale.add_argument(
        "--linear-omega",
        dest="log_omega",
        action="store_false",
        help="Plot omega on a linear y-axis (default).",
    )

    parser.add_argument(
        "--no-fixed-eta",
        dest="include_eta",
        action="store_false",
        default=True,
        help="Do not add fixed eta=1.0 when eta is absent from the scalar CSV.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    if args.folder is not None:
        output_folder = args.output_prefix or args.folder
        os.makedirs(output_folder, exist_ok=True)
        for stem, omega_path, scalar_path in find_csv_pairs(args.folder):
            stem_name = os.path.basename(stem)
            omega_out = os.path.join(output_folder, stem_name + "_omega_plot.png")
            scalar_out = os.path.join(output_folder, stem_name + "_scalar_plot.png")
            write_plots(
                omega_path,
                scalar_path,
                omega_out,
                scalar_out,
                log_omega=args.log_omega,
                include_eta=args.include_eta,
            )
    else:
        output_prefix = args.output_prefix or args.stem
        omega_path = args.stem + "_omega.csv"
        scalar_path = args.stem + "_scalar.csv"
        omega_out = output_prefix + "_omega_plot.png"
        scalar_out = output_prefix + "_scalar_plot.png"
        write_plots(
            omega_path,
            scalar_path,
            omega_out,
            scalar_out,
            log_omega=args.log_omega,
            include_eta=args.include_eta,
        )


if __name__ == "__main__":
    main()
