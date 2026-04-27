#!/usr/bin/env python
"""Compare per-site omega estimates from TOMBOMBADIL (MAP) and a Stan MCMC fit.

Reads a CmdStanMCMC RDS file produced by R/Stan, extracts omega posterior
summaries via an Rscript subprocess, then runs the TOMBOMBADIL gradient
optimizer on the same alignment. Produces a two-page PDF:

  Page 1 — Per-site comparison: Stan median + 95 % CI (shaded) vs. TOMBOMBADIL MAP
  Page 2 — Scatter: TOMBOMBADIL MAP (x) vs. Stan median (y), log-log, identity line

Usage:
    python compare_to_stan.py \\
        --alignment data/lamB_revtrans.fas.aln \\
        --rds data/fit_full_lamB.RDS \\
        [--iter 500] \\
        [--output comparison.pdf]
"""

import argparse
import logging
import os
import subprocess
import sys
import tempfile

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from matplotlib.backends.backend_pdf import PdfPages

from tombombadil.__main__ import count_codons
from tombombadil.sample import (
    _run_replicates,
    make_fn,
    positive,
    softplus_inverse,
    transforms,
)


def load_stan_omega(rds_path: str):
    """Extract per-site omega posterior summaries from a CmdStanMCMC RDS file.

    Uses an Rscript subprocess to read the file and write omega summaries to a
    temp CSV (mean, median, 2.5 %, 97.5 % quantile per site).

    Args:
        rds_path: absolute or relative path to the .RDS file

    Returns:
        omega_mean, omega_median, omega_lo95, omega_hi95 — each a float64
        array of shape (n_sites,), ordered by site index (omega[1] … omega[N])
    """
    rscript = _find_rscript()
    rds_abs = os.path.abspath(rds_path)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name

    with tempfile.NamedTemporaryFile(suffix=".R", delete=False, mode="w") as rtmp:
        rtmp_path = rtmp.name
        rtmp.write(f"""
suppressPackageStartupMessages(library(posterior))
fit  <- readRDS("{rds_abs}")
drws <- fit$draws(format = "df")
omega_cols <- sort(colnames(drws)[grepl("^omega[[]", colnames(drws))])
mat <- as.matrix(drws[omega_cols])
out <- data.frame(
    site   = seq_len(ncol(mat)),
    mean   = colMeans(mat),
    median = apply(mat, 2, median),
    lo95   = apply(mat, 2, quantile, 0.025),
    hi95   = apply(mat, 2, quantile, 0.975)
)
write.csv(out, "{tmp_path}", row.names = FALSE, quote = FALSE)
""")

    try:
        result = subprocess.run(
            [rscript, "--vanilla", rtmp_path],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as exc:
        logging.error("Rscript failed:\n%s", exc.stderr)
        os.unlink(tmp_path)
        raise RuntimeError("Failed to extract omega summaries from RDS — see stderr above.") from exc
    finally:
        os.unlink(rtmp_path)

    try:
        data = np.genfromtxt(tmp_path, delimiter=",", names=True)
    finally:
        os.unlink(tmp_path)

    return (
        data["mean"].astype(np.float64),
        data["median"].astype(np.float64),
        data["lo95"].astype(np.float64),
        data["hi95"].astype(np.float64),
    )


def _find_rscript() -> str:
    """Return the path to Rscript, or raise if not found."""
    for candidate in ["Rscript", "/usr/local/bin/Rscript", "/usr/bin/Rscript"]:
        if _cmd_exists(candidate):
            return candidate
    raise FileNotFoundError(
        "Rscript not found. Install R and ensure Rscript is on PATH."
    )


def _cmd_exists(cmd: str) -> bool:
    try:
        subprocess.run([cmd, "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def fit_tombombadil(X, pi_eq, n_iter: int = 500) -> np.ndarray:
    """Run the TOMBOMBADIL gradient optimizer and return MAP omega estimates.

    Args:
        X:      (61, n_sites) codon count matrix from count_codons()
        pi_eq:  (61,) equilibrium frequency vector
        n_iter: number of optimizer iterations

    Returns:
        omega_map: float64 array of shape (n_sites,) on the natural scale
    """
    log_pi, pimat, pimatinv, pimult = transforms(X, pi_eq)

    col_max = np.max(X, axis=0)
    col_sum = np.sum(X, axis=0)
    mask = np.where(col_max == col_sum, 0, 1)

    base_params = {
        "alpha":   jnp.array(softplus_inverse(1),   dtype=jnp.float64),
        "beta":    jnp.array(softplus_inverse(1),   dtype=jnp.float64),
        "gamma":   jnp.array(softplus_inverse(1),   dtype=jnp.float64),
        "delta":   jnp.array(softplus_inverse(1),   dtype=jnp.float64),
        "epsilon": jnp.array(softplus_inverse(1),   dtype=jnp.float64),
        "eta":     jnp.array(softplus_inverse(1),   dtype=jnp.float64),
        "theta":   jnp.array(softplus_inverse(0.5), dtype=jnp.float64),
        "omega":   jnp.repeat(
            jnp.array(softplus_inverse(0.5), dtype=jnp.float64),
            jnp.size(X, axis=1),
        ),
    }
    base_labels = {
        "omega": "vec", "alpha": "scalar", "beta": "scalar", "gamma": "scalar",
        "delta": "scalar", "epsilon": "scalar", "eta": "scalar", "theta": "scalar",
    }

    fn = make_fn(pi_eq, log_pi, pimat, pimatinv, pimult, X, mask)
    all_params, best_idx = _run_replicates(fn, base_params, base_labels, n_reps=1, n_iter=n_iter)
    return np.array(positive(all_params[best_idx]["omega"]))


def plot_per_site(omega_map, omega_median, omega_lo, omega_hi):
    """Per-site comparison: Stan 95 % CI ribbon + median line vs. TOMBOMBADIL MAP dots."""
    n = len(omega_map)
    sites = np.arange(n)

    fig, ax = plt.subplots(figsize=(16, 4))

    ax.fill_between(sites, omega_lo, omega_hi,
                    color="steelblue", alpha=0.25, label="Stan 95 % CI")
    ax.plot(sites, omega_median, color="steelblue", linewidth=0.8, label="Stan posterior median")
    ax.scatter(sites, omega_map, color="black", s=6, alpha=0.6, linewidths=0,
               label="TOMBOMBADIL MAP", zorder=4)

    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2, alpha=0.8, label="ω = 1 (neutral)")
    ax.set_yscale("log")
    ax.set_xlabel("Alignment site index")
    ax.set_ylabel("ω (dN/dS, log scale)")
    ax.set_title("Per-site ω: TOMBOMBADIL MAP vs. Stan MCMC")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=8)
    ax.set_xlim(-1, n)
    plt.tight_layout()
    return fig


def plot_scatter(omega_map, omega_median, omega_lo, omega_hi):
    """Scatter: TOMBOMBADIL MAP (x) vs. Stan posterior median (y), log-log with identity line."""
    fig, ax = plt.subplots(figsize=(6, 6))

    err_lo = omega_median - omega_lo
    err_hi = omega_hi - omega_median
    ax.errorbar(
        omega_map, omega_median,
        yerr=[err_lo, err_hi],
        fmt="o", color="steelblue", alpha=0.4, markersize=3,
        elinewidth=0.5, capsize=0, label="Sites (Stan 95 % CI)",
    )

    lo = min(omega_map.min(), omega_median.min()) * 0.9
    hi = max(omega_map.max(), omega_median.max()) * 1.1
    lo = max(lo, 1e-4)
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0, label="Identity (y = x)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("TOMBOMBADIL MAP ω")
    ax.set_ylabel("Stan posterior median ω")
    ax.set_title("TOMBOMBADIL vs. Stan — per-site ω (log scale)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def main():
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Compare TOMBOMBADIL MAP omega to Stan MCMC estimates."
    )
    parser.add_argument("--alignment", required=True,
                        help="Codon alignment FASTA (plain or gzipped)")
    parser.add_argument("--rds", required=True,
                        help="CmdStanMCMC RDS file produced by R/Stan")
    parser.add_argument("--iter", type=int, default=500,
                        help="TOMBOMBADIL optimizer iterations (default: 500)")
    parser.add_argument("--output", default="comparison.pdf",
                        help="Output PDF (default: comparison.pdf)")
    args = parser.parse_args()

    logging.info("Reading alignment: %s", args.alignment)
    X, n_samples = count_codons(args.alignment)
    n_sites = X.shape[1]
    logging.info("  %d sequences, %d codon sites", n_samples, n_sites)

    logging.info("Extracting Stan omega summaries from: %s", args.rds)
    omega_mean, omega_median, omega_lo, omega_hi = load_stan_omega(args.rds)
    logging.info("  %d Stan omega sites loaded", len(omega_median))

    if len(omega_median) != n_sites:
        logging.warning(
            "Stan has %d omega sites but alignment has %d codon columns — "
            "proceeding with min(%d, %d) sites for comparison.",
            len(omega_median), n_sites, len(omega_median), n_sites,
        )
        n = min(len(omega_median), n_sites)
        omega_mean, omega_median, omega_lo, omega_hi = (
            omega_mean[:n], omega_median[:n], omega_lo[:n], omega_hi[:n]
        )
        X = X[:, :n]

    pi_eq = np.array([1 / 61] * 61)

    logging.info("Fitting TOMBOMBADIL (MAP, %d iterations)...", args.iter)
    omega_map = fit_tombombadil(X, pi_eq, n_iter=args.iter)

    logging.info("Writing plots to: %s", args.output)
    with PdfPages(args.output) as pdf:
        fig1 = plot_per_site(omega_map, omega_median, omega_lo, omega_hi)
        pdf.savefig(fig1)
        plt.close(fig1)

        fig2 = plot_scatter(omega_map, omega_median, omega_lo, omega_hi)
        pdf.savefig(fig2)
        plt.close(fig2)

    logging.info("Done — %s", args.output)


if __name__ == "__main__":
    main()
