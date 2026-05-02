#!/usr/bin/env python
"""Check convergence of TOMBOMBADIL MAP runs against Stan MCMC.

Reads multiple JAX output CSV pairs (STEM_omega.csv + STEM_scalar.csv) and
a Stan RDS file, then produces a two-page PDF:

  Page 1 — Per-site omega (linear scale): Stan 95 % CI ribbon + median,
            one scatter per JAX run coloured by step count (light → dark).
            Optional domain colouring via marker shape (circle = other,
            x = extracellular).
  Page 2 — Scalar parameter forest plot: Stan posterior CI per parameter,
            JAX MAP estimates per run as dots coloured by step count.

Step counts are inferred from the stem filename, e.g. data/jax_fit_50 → 50.

Usage:
    python check_convergence.py \\
        --jax-runs data/jax_fit_50 data/jax_fit_200 data/jax_fit_500 \\
        --rds data/fit_full_lamB.RDS \\
        [--alignment data/lamB_revtrans.fas.aln] \\
        [--domain-json data/lamB_domains.JSON] \\
        [--reference-protein data/lamB_reference.fas] \\
        [--output convergence.pdf]
"""

import argparse
import csv
import logging
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from compare_to_stan import load_stan_omega, load_stan_scalar_params

_SCALAR_PARAMS = ["alpha", "beta", "gamma", "delta", "epsilon", "eta", "theta"]


def _steps_from_stem(stem: str) -> int:
    """Extract trailing step count from a stem like 'data/jax_fit_50' → 50."""
    m = re.search(r'_(\d+)$', stem)
    return int(m.group(1)) if m else -1


def load_jax_run(stem: str):
    """Load a JAX output pair and return (n_steps, omega_array, scalar_dict).

    Reads STEM_omega.csv and STEM_scalar.csv. eta is always added as 1.0
    (fixed in JAX, not written to CSV) so the scalar dict matches Stan's keys.
    """
    omega_data = np.genfromtxt(stem + "_omega.csv", delimiter=",", names=True)
    omega = omega_data["omega_map"].astype(np.float64)

    scalar_dict = {}
    with open(stem + "_scalar.csv") as f:
        for row in csv.DictReader(f):
            scalar_dict[row["variable"]] = float(row["value"])
    scalar_dict["eta"] = 1.0

    return _steps_from_stem(stem), omega, scalar_dict


def _step_colours(step_counts):
    """Map step counts to a light-to-dark plasma colour ramp."""
    cmap = plt.cm.plasma
    order = np.argsort(step_counts)
    n = len(step_counts)
    positions = np.empty(n)
    for rank, idx in enumerate(order):
        positions[idx] = 0.15 + 0.70 * (rank / max(n - 1, 1))
    return [cmap(p) for p in positions]


def plot_omega_convergence(jax_runs, stan_median, stan_lo, stan_hi,
                           is_extracellular=None, is_imputed=None, regression_mask=None):
    """Per-site omega (linear y-axis): Stan CI + one scatter per JAX run.

    Colour encodes step count (light = few steps, dark = many steps).
    When domain annotations are provided, extracellular sites use an 'x'
    marker and other sites use circles; NA/unannotated sites are plotted
    as small faint dots.
    """
    n = len(stan_median)
    sites = np.arange(n)
    use_domains = is_extracellular is not None

    fig, ax = plt.subplots(figsize=(16, 4))

    ax.fill_between(sites, stan_lo, stan_hi, color="grey", alpha=0.2)
    ax.plot(sites, stan_median, color="grey", linewidth=0.8)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.0, alpha=0.7)

    step_counts = [r[0] for r in jax_runs]
    colours = _step_colours(step_counts)

    if use_domains:
        is_ext  = np.array(is_extracellular, dtype=bool)
        is_imp  = np.array(is_imputed,       dtype=bool)
        reg_m   = np.array(regression_mask,  dtype=bool)
        ext_mask   = is_ext  & ~is_imp
        other_mask = ~is_ext & ~is_imp & reg_m
        na_mask    = ~reg_m

    for (n_steps, omega, _), colour in zip(reversed(jax_runs), reversed(colours)):
        if use_domains:
            ax.scatter(sites[other_mask], omega[other_mask],
                       color=colour, marker="o", s=5, alpha=0.4, linewidths=0, zorder=4)
            ax.scatter(sites[ext_mask], omega[ext_mask],
                       color=colour, marker="x", s=8, alpha=0.5, linewidths=0.7, zorder=4)
            if na_mask.any():
                ax.scatter(sites[na_mask], omega[na_mask],
                           color=colour, marker=".", s=3, alpha=0.2, linewidths=0, zorder=3)
        else:
            ax.scatter(sites, omega, color=colour, s=5, alpha=0.4,
                       linewidths=0, zorder=4)

    # Build legend manually so it is always complete regardless of domain masks
    legend_entries = [
        plt.Rectangle((0, 0), 1, 1, fc="grey", alpha=0.35, label="Stan 95 % CI"),
        Line2D([0], [0], color="grey",  linewidth=0.8,   label="Stan posterior median"),
        Line2D([0], [0], color="red",   linewidth=1.0,   linestyle="--", label="ω = 1"),
    ]
    for (n_steps, _, _), colour in zip(jax_runs, colours):
        legend_entries.append(
            Line2D([0], [0], marker="o", color=colour, linestyle="None",
                   markersize=5, label=f"JAX {n_steps} steps")
        )
    if use_domains:
        legend_entries += [
            Line2D([0], [0], marker="o", color="grey", linestyle="None",
                   markersize=5, label="Other (annotated)"),
            Line2D([0], [0], marker="x", color="grey", linestyle="None",
                   markersize=5, markeredgewidth=0.8, label="Extracellular (annotated)"),
        ]

    ax.legend(handles=legend_entries, loc="upper left", bbox_to_anchor=(1.01, 1),
              borderaxespad=0, fontsize=8)
    ax.set_xlabel("Alignment site index")
    ax.set_ylabel("ω (dN/dS)")
    ax.set_title("Per-site ω: TOMBOMBADIL MAP convergence vs. Stan MCMC (linear scale)")
    ax.set_xlim(-1, n)
    plt.tight_layout()
    return fig


def plot_params_convergence(jax_runs, stan_params):
    """Forest plot: Stan posterior CI + JAX MAP dots per run per parameter.

    Parameters on y-axis (log-scaled x). All JAX dots sit at the same y position
    as the Stan estimate; runs with fewer steps are plotted on top. Colour encodes
    step count (light = few steps, dark = many steps).
    eta is shown with JAX's fixed value (1.0) as a square marker vs. Stan's
    posterior, which reflects the unidentified GTR rate scale.
    """
    # Normalise Stan eta to eta/eta = 1 (a fixed point) so it is directly
    # comparable to JAX, which also fixes eta = 1 to identify the GTR rate scale.
    stan_params = dict(stan_params)
    if "eta" in stan_params:
        stan_params["eta"] = {"mean": 1.0, "median": 1.0, "lo95": 1.0, "hi95": 1.0}

    params = [p for p in _SCALAR_PARAMS if p in stan_params]
    y = np.arange(len(params))
    n_runs = len(jax_runs)

    step_counts = [r[0] for r in jax_runs]
    colours = _step_colours(step_counts)

    fig, ax = plt.subplots(figsize=(7.5, len(params) * 0.7 + 2.0))

    for i, name in enumerate(params):
        s = stan_params[name]
        ax.plot([s["lo95"], s["hi95"]], [i, i], color="steelblue",
                linewidth=2.5, alpha=0.5, zorder=2,
                label="Stan 95 % CI" if i == 0 else None)
        ax.scatter(s["median"], i, color="steelblue", s=55, zorder=3,
                   label="Stan posterior median" if i == 0 else None)

    for (n_steps, _, scalar_dict), colour in zip(reversed(jax_runs), reversed(colours)):
        first = True
        for i, name in enumerate(params):
            if name not in scalar_dict:
                continue
            marker = "s" if name == "eta" else "D"
            ax.scatter(scalar_dict[name], i,
                       color=colour, s=45, marker=marker, alpha=0.6,
                       edgecolors="black", linewidths=0.4, zorder=4,
                       label=f"JAX {n_steps} steps" if first else None)
            first = False

    ax.set_yticks(y)
    ax.set_yticklabels(params)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Parameter value (log scale)")
    ax.set_title(
        "Scalar parameters: convergence across JAX runs vs. Stan MCMC\n"
        "Stan GTR rates shown as x/eta; eta shown as eta/eta = 1 for both Stan and JAX"
    )
    ax.axvline(1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    return fig


def main():
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Check convergence of TOMBOMBADIL MAP runs against Stan MCMC."
    )
    parser.add_argument("--jax-runs", nargs="+", required=True, metavar="STEM",
                        help="One or more JAX output stems (e.g. data/jax_fit_50). "
                             "Reads STEM_omega.csv and STEM_scalar.csv for each.")
    parser.add_argument("--rds", required=True,
                        help="CmdStanMCMC RDS file produced by R/Stan")
    parser.add_argument("--output", default="convergence.pdf",
                        help="Output PDF (default: convergence.pdf)")
    parser.add_argument("--domain-json", default=None,
                        help="UniProt JSON with domain annotations (optional)")
    parser.add_argument("--reference-protein", default=None,
                        help="Reference protein FASTA (required with --domain-json)")
    parser.add_argument("--alignment", default=None,
                        help="Codon alignment FASTA (required with --domain-json)")
    args = parser.parse_args()

    use_domains = args.domain_json is not None
    if use_domains and (args.reference_protein is None or args.alignment is None):
        parser.error("--alignment and --reference-protein are both required with --domain-json")

    logging.info("Loading Stan omega summaries from: %s", args.rds)
    _, omega_median, omega_lo, omega_hi = load_stan_omega(args.rds)
    n_sites = len(omega_median)
    logging.info("  %d Stan omega sites", n_sites)

    logging.info("Loading Stan scalar parameter summaries...")
    stan_scalar_params = load_stan_scalar_params(args.rds)

    logging.info("Loading JAX runs...")
    jax_runs = []
    for stem in args.jax_runs:
        n_steps, omega, scalar_dict = load_jax_run(stem)
        if len(omega) != n_sites:
            logging.warning(
                "  '%s' has %d sites but Stan has %d — truncating to min",
                stem, len(omega), n_sites,
            )
            omega = omega[:min(len(omega), n_sites)]
        logging.info("  %s — %d steps, %d sites", stem, n_steps, len(omega))
        jax_runs.append((n_steps, omega, scalar_dict))
    jax_runs.sort(key=lambda r: r[0])

    is_extracellular = is_imputed = regression_mask = None
    if use_domains:
        from tombombadil.__main__ import count_codons
        from tombombadil.domains import parse_domain_json
        logging.info("Parsing domain annotations from: %s", args.domain_json)
        X, _ = count_codons(args.alignment)
        is_extracellular, is_imputed, regression_mask = parse_domain_json(
            args.domain_json, args.alignment, args.reference_protein, X.shape[1]
        )

    logging.info("Writing plots to: %s", args.output)
    with PdfPages(args.output) as pdf:
        fig1 = plot_omega_convergence(
            jax_runs, omega_median, omega_lo, omega_hi,
            is_extracellular, is_imputed, regression_mask,
        )
        pdf.savefig(fig1)
        plt.close(fig1)

        fig2 = plot_params_convergence(jax_runs, stan_scalar_params)
        pdf.savefig(fig2)
        plt.close(fig2)

    logging.info("Done — %s", args.output)


if __name__ == "__main__":
    main()
