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
import csv
import json
import logging
import os
import re
import subprocess
import sys
import tempfile

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from matplotlib.backends.backend_pdf import PdfPages

from tombombadil.__main__ import count_codons
from tombombadil.domains import parse_domain_json
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
    CSV alongside the RDS file in the same directory (mean, median, 2.5 %,
    97.5 % quantile per site). The CSV is kept after the run.

    Args:
        rds_path: absolute or relative path to the .RDS file

    Returns:
        omega_mean, omega_median, omega_lo95, omega_hi95 — each a float64
        array of shape (n_sites,), ordered by site index (omega[1] … omega[N])
    """
    rscript = _find_rscript()
    rds_abs = os.path.abspath(rds_path)

    rds_stem = os.path.splitext(rds_abs)[0]
    csv_path = rds_stem + "_omega_summaries.csv"

    with tempfile.NamedTemporaryFile(suffix=".R", delete=False, mode="w") as rtmp:
        rtmp_path = rtmp.name
        rtmp.write(f"""
suppressPackageStartupMessages(library(posterior))
fit  <- readRDS("{rds_abs}")
drws <- fit$draws(variables = "omega")
summ <- posterior::summarise_draws(
    drws,
    mean   = mean,
    median = median,
    lo95   = ~quantile(.x, 0.025)[[1]],
    hi95   = ~quantile(.x, 0.975)[[1]]
)
out <- data.frame(
    site   = seq_len(nrow(summ)),
    mean   = summ$mean,
    median = summ$median,
    lo95   = summ$lo95,
    hi95   = summ$hi95
)
write.csv(out, "{csv_path}", row.names = FALSE, quote = FALSE)
""")

    try:
        subprocess.run(
            [rscript, "--vanilla", rtmp_path],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as exc:
        logging.error("Rscript failed:\n%s", exc.stderr)
        raise RuntimeError("Failed to extract omega summaries from RDS — see stderr above.") from exc
    finally:
        os.unlink(rtmp_path)

    logging.info("Stan omega summaries saved to: %s", csv_path)
    data = np.genfromtxt(csv_path, delimiter=",", names=True)

    return (
        data["mean"].astype(np.float64),
        data["median"].astype(np.float64),
        data["lo95"].astype(np.float64),
        data["hi95"].astype(np.float64),
    )


_SCALAR_PARAMS = ["alpha", "beta", "gamma", "delta", "epsilon", "eta", "theta"]


def load_stan_scalar_params(rds_path: str) -> dict:
    """Extract scalar GTR parameter posterior summaries from a CmdStanMCMC RDS file.

    Returns a dict mapping each parameter name to (mean, median, lo95, hi95).
    Results are saved to a CSV alongside the RDS file and kept after the run.
    """
    rscript = _find_rscript()
    rds_abs = os.path.abspath(rds_path)
    rds_stem = os.path.splitext(rds_abs)[0]
    csv_path = rds_stem + "_scalar_summaries.csv"

    with tempfile.NamedTemporaryFile(suffix=".R", delete=False, mode="w") as rtmp:
        rtmp_path = rtmp.name
        rtmp.write(f"""
suppressPackageStartupMessages(library(posterior))
fit  <- readRDS("{rds_abs}")
drws <- fit$draws()
summ <- posterior::summarise_draws(
    drws,
    mean   = mean,
    median = median,
    lo95   = ~quantile(.x, 0.025)[[1]],
    hi95   = ~quantile(.x, 0.975)[[1]]
)
out <- summ[!startsWith(summ$variable, "omega[") & summ$variable != "lp__", c("variable", "mean", "median", "lo95", "hi95")]
write.csv(out, "{csv_path}", row.names = FALSE)
""")

    try:
        subprocess.run(
            [rscript, "--vanilla", rtmp_path],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as exc:
        logging.error("Rscript failed:\n%s", exc.stderr)
        raise RuntimeError("Failed to extract scalar summaries from RDS — see stderr above.") from exc
    finally:
        os.unlink(rtmp_path)

    logging.info("Stan scalar summaries saved to: %s", csv_path)
    with open(csv_path, newline="") as f:
        result = {}
        for row in csv.DictReader(f):
            name = re.sub(r'\[1\]$', '', row["variable"])
            result[name] = {
                "mean":   float(row["mean"]),
                "median": float(row["median"]),
                "lo95":   float(row["lo95"]),
                "hi95":   float(row["hi95"]),
            }
        return result


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
    best = all_params[best_idx]
    omega_map = np.array(positive(best["omega"]))
    scalar_params = {p: float(positive(best[p])) for p in _SCALAR_PARAMS}
    return omega_map, scalar_params


def load_slac_significance(json_path: str, p_threshold: float = 0.05):
    """Extract per-site selection significance from a HyPhy SLAC JSON file.

    Uses the binomial p-values from the AVERAGED by-site MLE table:
      col 8 — P[dN/dS > 1] (positive selection)
      col 9 — P[dN/dS < 1] (negative selection)

    Returns:
        pos_selected: bool array, True where P[dN/dS > 1] < p_threshold
        neg_selected: bool array, True where P[dN/dS < 1] < p_threshold
    """
    with open(json_path) as f:
        data = json.load(f)
    rows = data["MLE"]["content"]["0"]["by-site"]["AVERAGED"]
    n = len(rows)
    pos_selected = np.zeros(n, dtype=bool)
    neg_selected = np.zeros(n, dtype=bool)
    for i, row in enumerate(rows):
        p_pos, p_neg = row[8], row[9]
        if p_pos is not None and p_pos < p_threshold:
            pos_selected[i] = True
        if p_neg is not None and p_neg < p_threshold:
            neg_selected[i] = True
    return pos_selected, neg_selected


def load_fubar_dnds(json_path: str) -> np.ndarray:
    """Extract per-site mean posterior dN/dS (beta/alpha) from a HyPhy FUBAR JSON file.

    Returns:
        omega: float64 array of shape (n_sites,), NaN where alpha is zero
    """
    with open(json_path) as f:
        data = json.load(f)
    rows = data["MLE"]["content"]["0"]
    return np.array([
        row[1] / row[0] if row[0] > 0 else np.nan
        for row in rows
    ])


def plot_per_site_slac_highlighted(omega_map, omega_median_stan, omega_lo_stan, omega_hi_stan,
                                    pos_selected, neg_selected,
                                    is_extracellular=None, is_imputed=None,
                                    regression_mask=None, log_scale=True,
                                    p_threshold=0.05):
    """Per-site comparison of TOMBOMBADIL MAP and Stan posterior, with SLAC significance rug.

    Significant sites (SLAC binomial p < p_threshold) are shown as coloured ticks
    at the bottom of the plot: orange for positive selection, blue for negative selection.
    """
    n = len(omega_map)
    sites = np.arange(n)

    fig, ax = plt.subplots(figsize=(16, 4))

    ax.fill_between(sites, omega_lo_stan, omega_hi_stan,
                    color="grey", alpha=0.2, label="Stan 95 % CI")
    ax.plot(sites, omega_median_stan, color="grey", linewidth=0.8, label="Stan posterior median")

    if is_extracellular is not None:
        is_ext  = np.array(is_extracellular, dtype=bool)
        is_imp  = np.array(is_imputed,       dtype=bool)
        reg_m   = np.array(regression_mask,  dtype=bool)
        known_ext   = is_ext  & ~is_imp
        known_other = ~is_ext & ~is_imp & reg_m
        imputed     = is_imp
        na_sites    = ~reg_m
        ax.scatter(sites[known_other], omega_map[known_other], color="steelblue", s=6, alpha=0.6,
                   linewidths=0, label="TOMBOMBADIL MAP (other)", zorder=4)
        ax.scatter(sites[known_ext],   omega_map[known_ext],   color="tomato",    s=6, alpha=0.8,
                   linewidths=0, label="TOMBOMBADIL MAP (extracellular)", zorder=4)
        ax.scatter(sites[imputed],     omega_map[imputed],     color="grey",      s=6, alpha=0.5,
                   linewidths=0, label="TOMBOMBADIL MAP (imputed)", zorder=3)
        if na_sites.any():
            ax.scatter(sites[na_sites], omega_map[na_sites],   color="lightgrey", s=4, alpha=0.4,
                       linewidths=0, label="TOMBOMBADIL MAP (unknown)", zorder=2)
    else:
        ax.scatter(sites, omega_map, color="black", s=6, alpha=0.6, linewidths=0,
                   label="TOMBOMBADIL MAP", zorder=4)

    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2, alpha=0.8, label="ω = 1 (neutral)")

    trans = ax.get_xaxis_transform()
    if pos_selected.any():
        ax.vlines(sites[pos_selected], 0, 0.04, transform=trans,
                  color="darkorange", linewidth=1.2, alpha=0.9,
                  label=f"SLAC positive selection (p<{p_threshold})", zorder=5)
    if neg_selected.any():
        ax.vlines(sites[neg_selected], 0, 0.04, transform=trans,
                  color="royalblue", linewidth=1.2, alpha=0.9,
                  label=f"SLAC negative selection (p<{p_threshold})", zorder=5)

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Alignment site index")
    ax.set_ylabel("ω (dN/dS, log scale)" if log_scale else "ω (dN/dS)")
    ax.set_title("Per-site ω: TOMBOMBADIL MAP vs. Stan MCMC (SLAC significant sites highlighted)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=8)
    ax.set_xlim(-1, n)
    plt.tight_layout()
    return fig


def plot_params_comparison(jax_params: dict, stan_params: dict):
    """Forest-plot comparison of scalar GTR parameters: Stan posterior vs. TOMBOMBADIL MAP."""
    params = [p for p in stan_params if p in jax_params]
    y = np.arange(len(params))

    fig, ax = plt.subplots(figsize=(7, len(params) * 0.6 + 1))

    for i, name in enumerate(params):
        s = stan_params[name]
        ax.plot([s["lo95"], s["hi95"]], [i, i], color="steelblue", linewidth=2, zorder=2)
        ax.scatter(s["median"], i, color="steelblue", s=40, zorder=3, label="Stan posterior median" if i == 0 else None)
        ax.scatter(jax_params[name], i, color="tomato", s=40, marker="D", zorder=4, label="TOMBOMBADIL MAP" if i == 0 else None)

    ax.set_yticks(y)
    ax.set_yticklabels(params)
    ax.set_xscale("log")
    ax.set_xlabel("Parameter value (log scale)")
    ax.set_title("Scalar GTR parameters: TOMBOMBADIL MAP vs. Stan MCMC")
    ax.axvline(1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def plot_per_site(omega_map, omega_median, omega_lo, omega_hi,
                  is_extracellular=None, is_imputed=None, regression_mask=None,
                  log_scale=True):
    """Per-site comparison: Stan 95 % CI ribbon + median line vs. TOMBOMBADIL MAP dots."""
    n = len(omega_map)
    sites = np.arange(n)

    fig, ax = plt.subplots(figsize=(16, 4))

    ax.fill_between(sites, omega_lo, omega_hi,
                    color="grey", alpha=0.25, label="Stan 95 % CI")
    ax.plot(sites, omega_median, color="grey", linewidth=0.8, label="Stan posterior median")

    if is_extracellular is not None:
        is_ext  = np.array(is_extracellular, dtype=bool)
        is_imp  = np.array(is_imputed,       dtype=bool)
        reg_m   = np.array(regression_mask,  dtype=bool)
        known_ext   = is_ext  & ~is_imp
        known_other = ~is_ext & ~is_imp & reg_m
        imputed     = is_imp
        na_sites    = ~reg_m
        ax.scatter(sites[known_other], omega_map[known_other], color="steelblue", s=6, alpha=0.6,
                   linewidths=0, label="TOMBOMBADIL MAP (other)", zorder=4)
        ax.scatter(sites[known_ext],   omega_map[known_ext],   color="tomato",    s=6, alpha=0.8,
                   linewidths=0, label="TOMBOMBADIL MAP (extracellular)", zorder=4)
        ax.scatter(sites[imputed],     omega_map[imputed],     color="grey",      s=6, alpha=0.5,
                   linewidths=0, label="TOMBOMBADIL MAP (imputed)", zorder=3)
        if na_sites.any():
            ax.scatter(sites[na_sites], omega_map[na_sites],   color="lightgrey", s=4, alpha=0.4,
                       linewidths=0, label="TOMBOMBADIL MAP (unknown)", zorder=2)
    else:
        ax.scatter(sites, omega_map, color="black", s=6, alpha=0.6, linewidths=0,
                   label="TOMBOMBADIL MAP", zorder=4)

    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2, alpha=0.8, label="ω = 1 (neutral)")
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Alignment site index")
    ax.set_ylabel("ω (dN/dS, log scale)" if log_scale else "ω (dN/dS)")
    ax.set_title("Per-site ω: TOMBOMBADIL MAP vs. Stan MCMC")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=8)
    ax.set_xlim(-1, n)
    plt.tight_layout()
    return fig


def plot_scatter(omega_map, omega_median, omega_lo, omega_hi,
                 is_extracellular=None, is_imputed=None, regression_mask=None):
    """Scatter: TOMBOMBADIL MAP (x) vs. Stan posterior median (y), log-log with identity line."""
    fig, ax = plt.subplots(figsize=(6, 6))

    err_lo = omega_median - omega_lo
    err_hi = omega_hi - omega_median

    if is_extracellular is not None:
        is_ext  = np.array(is_extracellular, dtype=bool)
        is_imp  = np.array(is_imputed,       dtype=bool)
        reg_m   = np.array(regression_mask,  dtype=bool)
        known_ext   = is_ext  & ~is_imp
        known_other = ~is_ext & ~is_imp & reg_m
        imputed     = is_imp
        na_sites    = ~reg_m
        for mask, colour, label in [
            (known_other, "steelblue", "Other (annotated)"),
            (known_ext,   "tomato",    "Extracellular (annotated)"),
            (imputed,     "grey",      "Imputed"),
        ]:
            if mask.any():
                ax.errorbar(
                    omega_map[mask], omega_median[mask],
                    yerr=[err_lo[mask], err_hi[mask]],
                    fmt="o", color=colour, alpha=0.5, markersize=3,
                    elinewidth=0.5, capsize=0, label=label,
                )
        if na_sites.any():
            ax.errorbar(
                omega_map[na_sites], omega_median[na_sites],
                yerr=[err_lo[na_sites], err_hi[na_sites]],
                fmt="o", color="lightgrey", alpha=0.3, markersize=2,
                elinewidth=0.3, capsize=0, label="Unknown",
            )
    else:
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
    parser.add_argument("--domain-json", default=None,
                        help="UniProt JSON with domain annotations (optional)")
    parser.add_argument("--reference-protein", default=None,
                        help="Reference protein FASTA for alignment-to-protein mapping (required with --domain-json)")
    parser.add_argument("--slac-json", default=None,
                        help="HyPhy SLAC JSON output to highlight significant sites (optional)")
    parser.add_argument("--slac-p-threshold", type=float, default=0.05,
                        help="P-value threshold for SLAC significance (default: 0.05)")
    args = parser.parse_args()

    logging.info("Reading alignment: %s", args.alignment)
    X, n_samples = count_codons(args.alignment)
    n_sites = X.shape[1]
    logging.info("  %d sequences, %d codon sites", n_samples, n_sites)

    logging.info("Extracting Stan omega summaries from: %s", args.rds)
    omega_mean, omega_median, omega_lo, omega_hi = load_stan_omega(args.rds)
    logging.info("  %d Stan omega sites loaded", len(omega_median))

    logging.info("Extracting Stan scalar parameter summaries from: %s", args.rds)
    stan_scalar_params = load_stan_scalar_params(args.rds)

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

    is_extracellular = is_imputed = regression_mask = None
    if args.domain_json:
        if not args.reference_protein:
            parser.error("--reference-protein is required when --domain-json is provided")
        logging.info("Parsing domain annotations from: %s", args.domain_json)
        is_extracellular, is_imputed, regression_mask = parse_domain_json(
            args.domain_json, args.alignment, args.reference_protein, n_sites
        )

    slac_pos = slac_neg = None
    if args.slac_json:
        logging.info("Loading SLAC significance from: %s", args.slac_json)
        slac_pos, slac_neg = load_slac_significance(args.slac_json, args.slac_p_threshold)
        logging.info("  %d positively selected, %d negatively selected sites (p<%s)",
                     int(slac_pos.sum()), int(slac_neg.sum()), args.slac_p_threshold)

    logging.info("Fitting TOMBOMBADIL (MAP, %d iterations)...", args.iter)
    omega_map, jax_scalar_params = fit_tombombadil(X, pi_eq, n_iter=args.iter)

    logging.info("Writing plots to: %s", args.output)
    with PdfPages(args.output) as pdf:
        fig1 = plot_per_site(omega_map, omega_median, omega_lo, omega_hi,
                             is_extracellular, is_imputed, regression_mask)
        pdf.savefig(fig1)
        plt.close(fig1)

        fig1b = plot_per_site(omega_map, omega_median, omega_lo, omega_hi,
                              is_extracellular, is_imputed, regression_mask, log_scale=False)
        pdf.savefig(fig1b)
        plt.close(fig1b)

        fig2 = plot_scatter(omega_map, omega_median, omega_lo, omega_hi,
                            is_extracellular, is_imputed, regression_mask)
        pdf.savefig(fig2)
        plt.close(fig2)

        fig3 = plot_params_comparison(jax_scalar_params, stan_scalar_params)
        pdf.savefig(fig3)
        plt.close(fig3)

        if slac_pos is not None:
            fig4 = plot_per_site_slac_highlighted(
                omega_map, omega_median, omega_lo, omega_hi, slac_pos, slac_neg,
                is_extracellular, is_imputed, regression_mask,
                p_threshold=args.slac_p_threshold,
            )
            pdf.savefig(fig4)
            plt.close(fig4)

            fig4b = plot_per_site_slac_highlighted(
                omega_map, omega_median, omega_lo, omega_hi, slac_pos, slac_neg,
                is_extracellular, is_imputed, regression_mask, log_scale=False,
                p_threshold=args.slac_p_threshold,
            )
            pdf.savefig(fig4b)
            plt.close(fig4b)

    logging.info("Done — %s", args.output)


if __name__ == "__main__":
    main()
