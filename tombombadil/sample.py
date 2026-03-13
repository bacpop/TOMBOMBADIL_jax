#!/usr/bin/env python

import logging
import numpy as np
import scipy
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import jax.scipy.special as special
from jax.scipy.special import gammaln
import blackjax
from jax import jit
from jax.flatten_util import ravel_pytree
jax.config.update('jax_enable_x64', True)
import matplotlib.pyplot as plt

from .gtr import build_GTR
from .likelihood import gen_alpha

@jit
def my_dirichlet_multinomial_logpmf(x, a):
    x = jnp.asarray(x, dtype=jnp.float64)
    a = jnp.asarray(a, dtype=jnp.float64)

    N = jnp.sum(x, axis=-1)
    a0 = jnp.sum(a, axis=-1)

    term1 = gammaln(N + 1) - jnp.sum(gammaln(x + 1), axis=-1)
    term2 = gammaln(a0) - gammaln(N + a0)
    term3 = jnp.sum(gammaln(x + a) - (gammaln(a)), axis=-1)

    return term1 + term2 + term3 # gives 1407.2288

# This version is adapted from the scipy implementation
def my_dirichlet_multinomial_logpmf_2(x, a):
    x = jnp.asarray(x)
    a = jnp.asarray(a)

    N = jnp.sum(x, axis=-1)
    a0 = jnp.sum(a, axis=-1)

    out = jnp.asarray(gammaln(a0) + gammaln(N + 1) - gammaln(N + a0))
    out += (gammaln(x + a) - (gammaln(a) + gammaln(x + 1))).sum(axis=-1)

    return out

@jit
def model(alpha, beta, gamma, delta, epsilon, eta, mu, omega, pi_eq, log_pi, pimat, pimatinv, pimult, obs_vec):
    A = build_GTR(alpha, beta, gamma, delta, epsilon, eta, 1, pimat, pimult)
    meanrate = -jnp.dot(jnp.diagonal(A), pi_eq)
    scale = (mu / 2.0) / meanrate

    A2 = gen_alpha(omega, A, pimat, pimult, pimatinv, scale)
    log_prob = my_dirichlet_multinomial_logpmf(obs_vec, A2)

    return special.logsumexp(log_prob + log_pi, axis=0)

def transforms(X, pi_eq):
    log_pi = np.log(pi_eq)
    pimat = np.diag(np.sqrt(pi_eq))
    pimatinv = np.diag(np.divide(1, np.sqrt(pi_eq)))

    pimult = np.zeros((61, 61))
    for j in range(61):
        for i in range(61):
            pimult[i, j] = np.sqrt(pi_eq[j] / pi_eq[i])

    return log_pi, pimat, pimatinv, pimult

def positive(a):
    eps = 1e-6
    return jnp.exp(a) + eps

def softplus_inverse(y, eps=1e-6):
    z = y - eps
    return jnp.log((z))

def make_fn(pi_eq, log_pi, pimat, pimatinv, pimult, X, mask):
    batched_loss = jax.vmap(
        model,
        in_axes=(None, None, None, None, None, None, None, 0, None, None, None, None, None, 1)
    )
    def f(raw_x):
        x = jax.tree.map(positive, raw_x)

        x["omega"] = jnp.where(
            mask == 1,
            x["omega"],
            jax.lax.stop_gradient(x["omega"])
        )

        x["omega"] = jnp.where(
            x["omega"] > 0.01,
            x["omega"],
            jax.lax.stop_gradient(x["omega"])
        )

        losses = batched_loss(x["alpha"], x["beta"], x["gamma"], x["delta"], x["epsilon"], x["eta"], x["theta"], x["omega"], pi_eq, log_pi, pimat, pimatinv, pimult, X)
        return jnp.mean(losses)
    return f


def _run_nuts_chain(fn, init_params, warmup_steps, sample_steps, rng_key):
    """Run a single NUTS chain with window adaptation warmup.

    Args:
        fn:            log-density function (higher = better)
        init_params:   starting parameter pytree (unconstrained space)
        warmup_steps:  number of adaptation steps for step size / mass matrix
        sample_steps:  number of posterior samples to draw
        rng_key:       JAX PRNG key

    Returns:
        samples: pytree with same structure as init_params; each leaf has an
                 extra leading axis of length sample_steps.
    """
    warmup_key, sample_key = jax.random.split(rng_key)

    warmup = blackjax.window_adaptation(blackjax.nuts, fn)
    (state, adapted_params), _ = warmup.run(warmup_key, init_params, num_steps=warmup_steps)

    nuts_kernel = blackjax.nuts(fn, **adapted_params)

    def one_step(state, key):
        state, _ = nuts_kernel.step(key, state)
        return state, state.position

    sample_keys = jax.random.split(sample_key, sample_steps)
    _, samples = jax.lax.scan(one_step, state, sample_keys)
    return samples


def _perturb_params(params, scale=0.5):
    """Add Normal(0, scale) noise to all parameters in raw (unconstrained) space."""
    key = jax.random.PRNGKey(int(np.random.randint(0, 2**31)))
    flat, unflatten = ravel_pytree(params)
    noise = jax.random.normal(key, flat.shape) * scale
    return unflatten(flat + noise)


def _run_nuts_chains(fn, init_params, warmup_steps, sample_steps, num_chains):
    """Run multiple independent NUTS chains from perturbed starting points.

    Chain 0 uses the unperturbed starting point; subsequent chains add
    Normal(0, 0.5) noise in raw parameter space.  The best chain is identified
    by the log-likelihood evaluated at its posterior mean.

    Returns:
        all_samples: list of sample pytrees, one per chain
        best_idx:    index of the chain with the highest posterior-mean log-likelihood
    """
    base_key = jax.random.PRNGKey(0)
    keys = jax.random.split(base_key, num_chains)

    all_samples = []
    all_mean_lls = []

    for i in range(num_chains):
        start = init_params if i == 0 else _perturb_params(init_params)
        logging.info(f"  Running chain {i + 1}/{num_chains}...")
        samples = _run_nuts_chain(fn, start, warmup_steps, sample_steps, keys[i])
        all_samples.append(samples)

        # Evaluate log-likelihood at the posterior mean of this chain
        mean_params = jax.tree.map(lambda x: jnp.mean(x, axis=0), samples)
        mean_ll = float(fn(mean_params))
        all_mean_lls.append(mean_ll)
        logging.info(f"  Chain {i + 1}: log-likelihood at posterior mean = {mean_ll:.4f}")

    best_idx = int(np.argmax(all_mean_lls))
    logging.info(f"Best chain: {best_idx + 1} (log-likelihood = {all_mean_lls[best_idx]:.4f})")
    return all_samples, best_idx


def posthoc_regression(omega_posterior_mean, is_extracellular, regression_mask):
    """OLS regression of log(posterior mean omega) on domain annotation.

    Fits:  log(omega_i) = alpha_reg + beta_reg * is_extracellular_i + epsilon_i
    over all sites where regression_mask == 1.

    Returns:
        alpha_reg:  float, intercept on log-omega scale
        beta_reg:   float, extracellular effect on log-omega scale
        log_sigma:  float, log of residual std
    """
    log_omega = np.log(np.array(omega_posterior_mean))
    is_ext = np.array(is_extracellular)
    mask = np.array(regression_mask, dtype=bool)

    y = log_omega[mask]
    x = is_ext[mask]
    X_mat = np.column_stack([np.ones_like(x), x])

    coeffs, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
    alpha_reg, beta_reg = float(coeffs[0]), float(coeffs[1])

    sigma = float(np.std(y - X_mat @ coeffs))
    log_sigma = float(np.log(max(sigma, 1e-6)))

    return alpha_reg, beta_reg, log_sigma


def _print_posterior_summary(samples):
    """Print posterior summary (natural scale, 95% credible intervals) from NUTS samples."""
    gtr_keys = ["alpha", "beta", "gamma", "delta", "epsilon", "eta", "theta"]

    print("\n--- Posterior summary (natural scale, 95% CI) ---")
    print("GTR / shared parameters:")
    for k in gtr_keys:
        if k in samples:
            vals = np.array(positive(samples[k]))  # shape (n_samples,)
            print(f"  {k:8s}: mean={vals.mean():.4f}  "
                  f"median={np.median(vals):.4f}  "
                  f"95% CI=[{np.quantile(vals, 0.025):.4f}, {np.quantile(vals, 0.975):.4f}]")

    omega_nat = np.array(positive(samples["omega"]))  # (n_samples, n_sites)
    omega_means = omega_nat.mean(axis=0)
    print(f"Omega (posterior means): mean={omega_means.mean():.4f}  "
          f"min={omega_means.min():.4f}  max={omega_means.max():.4f}")
    print("--------------------------------------------------\n")


def plot_chains(all_samples_list, best_idx):
    """Plot per-site omega posterior means and GTR estimates across NUTS chains.

    Two panels:
      - Top: posterior mean omega per site, one colour per chain; best chain
             is opaque and larger, others are semi-transparent.
      - Bottom: GTR scalar parameter posterior means as a strip plot across chains.

    Tight clustering across chains indicates good mixing and convergence.
    """
    n_chains = len(all_samples_list)
    all_omega_means = [
        np.array(positive(jnp.mean(s["omega"], axis=0))) for s in all_samples_list
    ]
    n_sites = len(all_omega_means[0])
    sites = np.arange(n_sites)
    gtr_keys = ["alpha", "beta", "gamma", "delta", "epsilon", "eta", "theta"]
    cmap = plt.cm.tab10

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    ax = axes[0]
    for i, omega_means in enumerate(all_omega_means):
        is_best = (i == best_idx)
        ax.scatter(sites, omega_means, color=cmap(i % 10),
                   alpha=0.9 if is_best else 0.25,
                   s=15 if is_best else 8,
                   zorder=4 if is_best else 2,
                   label=f'Chain {i + 1} (best)' if is_best else f'Chain {i + 1}')
    ax.axhline(1.0, color='black', linestyle='-', linewidth=2.5, alpha=0.85,
               label='ω = 1 (neutral)', zorder=5)
    ax.set_yscale('log')
    ax.set_xlabel('Alignment site index')
    ax.set_ylabel('ω posterior mean (log scale)')
    ax.set_title(f'Per-site ω posterior means across {n_chains} chains — best chain highlighted')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=8)

    ax2 = axes[1]
    x_pos = np.arange(len(gtr_keys))
    for i, samples in enumerate(all_samples_list):
        is_best = (i == best_idx)
        vals = [float(jnp.mean(positive(samples[k]))) for k in gtr_keys if k in samples]
        ax2.scatter(x_pos[:len(vals)], vals, color=cmap(i % 10),
                    alpha=0.9 if is_best else 0.35,
                    s=60 if is_best else 25,
                    zorder=4 if is_best else 2)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(gtr_keys)
    ax2.axhline(1.0, color='black', linestyle='--', linewidth=1.0, alpha=0.5)
    ax2.set_ylabel('Posterior mean (natural scale)')
    ax2.set_title('GTR parameter posterior means across chains')

    plt.tight_layout()
    return fig, axes


def plot_regression(params, is_extracellular, is_imputed, regression_mask):
    """Plot per-site omega estimates coloured by domain type with regression predictions.

    Known annotated sites are shown in red/blue. Imputed sites are shown in grey.
    NA sites (excluded from regression) are shown in light grey. omega=1 is marked
    with a prominent horizontal line.
    """
    omega = np.array(positive(params["omega"]))
    alpha_reg = float(params["alpha_reg"])
    beta_reg  = float(params["beta_reg"])
    sigma     = float(jnp.exp(params["log_sigma"]))

    is_ext  = np.array(is_extracellular, dtype=bool)
    is_imp  = np.array(is_imputed,       dtype=bool)
    reg_m   = np.array(regression_mask,  dtype=bool)

    known_ext   = is_ext  & ~is_imp
    known_other = ~is_ext & ~is_imp & reg_m
    imputed     = is_imp
    na_sites    = ~reg_m

    sites = np.arange(len(omega))
    fig, ax = plt.subplots(figsize=(14, 4))

    ax.scatter(sites[known_other], omega[known_other], color='steelblue', alpha=0.5, s=15, label='Other (annotated)',         zorder=3)
    ax.scatter(sites[known_ext],   omega[known_ext],   color='tomato',    alpha=0.7, s=15, label='Extracellular (annotated)', zorder=3)
    ax.scatter(sites[imputed],     omega[imputed],     color='grey',      alpha=0.5, s=15, label='Imputed',                  zorder=3)
    if na_sites.any():
        ax.scatter(sites[na_sites], omega[na_sites],   color='lightgrey', alpha=0.4, s=10, label='Unknown (excluded)',       zorder=2)

    pred_other = np.exp(alpha_reg)
    pred_ext   = np.exp(alpha_reg + beta_reg)
    for pred, colour, label in [
        (pred_other, 'steelblue', f'E[ω | other] = {pred_other:.3f}'),
        (pred_ext,   'tomato',    f'E[ω | extracellular] = {pred_ext:.3f}'),
    ]:
        ax.axhline(pred,                          color=colour, linestyle='--', linewidth=1.5,            label=label)
        ax.axhline(np.exp(np.log(pred) + sigma),  color=colour, linestyle=':',  linewidth=0.8, alpha=0.5)
        ax.axhline(np.exp(np.log(pred) - sigma),  color=colour, linestyle=':',  linewidth=0.8, alpha=0.5,
                   label=f'±1σ (σ={sigma:.2f})')

    ax.axhline(1.0, color='black', linestyle='-', linewidth=2.5, alpha=0.85,
               label='ω = 1 (neutral)', zorder=5)

    ax.set_yscale('log')
    ax.set_xlabel('Alignment site index')
    ax.set_ylabel('ω (dN/dS, log scale)')
    ax.set_title(
        f'Per-site ω with posthoc domain regression\n'
        f'α={alpha_reg:.3f}, β={beta_reg:.3f}, σ={sigma:.3f}'
    )
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), loc='upper left', bbox_to_anchor=(1.01, 1),
              borderaxespad=0, fontsize=8)
    plt.tight_layout()
    return fig, ax


def plot_omega_coloured(params, is_extracellular, is_imputed, regression_mask):
    """Plot per-site omega estimates coloured by domain type, without regression lines."""
    omega = np.array(positive(params["omega"]))

    is_ext  = np.array(is_extracellular, dtype=bool)
    is_imp  = np.array(is_imputed,       dtype=bool)
    reg_m   = np.array(regression_mask,  dtype=bool)

    known_ext   = is_ext  & ~is_imp
    known_other = ~is_ext & ~is_imp & reg_m
    imputed     = is_imp
    na_sites    = ~reg_m

    sites = np.arange(len(omega))
    fig, ax = plt.subplots(figsize=(14, 4))

    ax.scatter(sites[known_other], omega[known_other], color='steelblue', alpha=0.5, s=15, label='Other (annotated)',         zorder=3)
    ax.scatter(sites[known_ext],   omega[known_ext],   color='tomato',    alpha=0.7, s=15, label='Extracellular (annotated)', zorder=3)
    ax.scatter(sites[imputed],     omega[imputed],     color='grey',      alpha=0.5, s=15, label='Imputed',                  zorder=3)
    if na_sites.any():
        ax.scatter(sites[na_sites], omega[na_sites],   color='lightgrey', alpha=0.4, s=10, label='Unknown (excluded)',       zorder=2)

    ax.axhline(1.0, color='black', linestyle='-', linewidth=2.5, alpha=0.85,
               label='ω = 1 (neutral)', zorder=5)

    ax.set_yscale('log')
    ax.set_xlabel('Alignment site index')
    ax.set_ylabel('ω (dN/dS, log scale)')
    ax.set_title('Per-site ω coloured by domain annotation (no regression)')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=8)
    plt.tight_layout()
    return fig, ax


def plot_omega_by_domain(params, is_extracellular, is_imputed, regression_mask,
                         diversity_mask=None):
    """Overlapping histogram comparing omega distributions for extracellular vs other sites.

    Omega appears on the x-axis (log scale); the y-axis shows proportion (density).
    Imputed, unannotated, and invariant (no-diversity) sites are excluded so only
    directly-annotated, variable sites are compared.  The omega=1 neutral line is
    shown for reference.
    """
    omega   = np.array(positive(params["omega"]))
    is_ext  = np.array(is_extracellular, dtype=bool)
    is_imp  = np.array(is_imputed,       dtype=bool)
    reg_m   = np.array(regression_mask,  dtype=bool)
    div_m   = np.array(diversity_mask, dtype=bool) if diversity_mask is not None \
              else np.ones(len(omega), dtype=bool)

    known_ext   = is_ext  & ~is_imp & div_m
    known_other = ~is_ext & ~is_imp & reg_m & div_m

    groups  = ["Other", "Extracellular"]
    colours = {"Extracellular": "tomato", "Other": "steelblue"}
    masks   = {"Extracellular": known_ext, "Other": known_other}

    all_vals = omega[known_ext | known_other]
    lo = np.log10(max(all_vals.min(), 1e-3))
    hi = np.log10(all_vals.max() * 1.05)
    bins = np.logspace(lo, hi, 30)

    fig, ax = plt.subplots(figsize=(6, 4))

    for group in groups:
        vals = omega[masks[group]]
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=bins, density=True, color=colours[group],
                alpha=0.4, label=f'{group} (n={len(vals)})', zorder=3)
        ax.hist(vals, bins=bins, density=True, color=colours[group],
                histtype='step', linewidth=1.2, zorder=4)

    ax.axvline(1.0, color='black', linestyle='-', linewidth=2, alpha=0.8,
               label='ω = 1 (neutral)', zorder=5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('ω (dN/dS, log scale)')
    ax.set_ylabel('Proportion (log scale)')
    ax.set_title('ω distribution by domain annotation\n(annotated, variable sites only)')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=8)
    plt.tight_layout()
    return fig, ax


def run_sampler(X, pi_eq, warmup=100, samples=100, platform='cpu', threads=8,
                is_extracellular=None, is_imputed=None, regression_mask=None,
                only_colour_domains=False, estimate_uncertainty=False, fit_replicates=1):
    logging.info("Precomputing transforms...")
    log_pi, pimat, pimatinv, pimult = transforms(X, pi_eq)

    col_max = np.max(X, axis=0)
    col_sum = np.sum(X, axis=0)
    mask = np.where(col_max == col_sum, 0, 1)

    logging.info("Compiling model...")

    base_params = {
        "alpha":   jnp.array(softplus_inverse(1),   dtype=jnp.float64),
        "beta":    jnp.array(softplus_inverse(1),   dtype=jnp.float64),
        "gamma":   jnp.array(softplus_inverse(1),   dtype=jnp.float64),
        "delta":   jnp.array(softplus_inverse(1),   dtype=jnp.float64),
        "epsilon": jnp.array(softplus_inverse(1),   dtype=jnp.float64),
        "eta":     jnp.array(softplus_inverse(1),   dtype=jnp.float64),
        "theta":   jnp.array(softplus_inverse(0.5), dtype=jnp.float64),
        "omega":   jnp.repeat(jnp.array(softplus_inverse(0.5), dtype=jnp.float64), jnp.size(X, axis=1)),
    }

    fn = make_fn(pi_eq, log_pi, pimat, pimatinv, pimult, X, mask)

    if only_colour_domains and is_extracellular is not None:
        logging.info(f"Running NUTS sampling ({fit_replicates} chain(s), {warmup} warmup / {samples} sample steps)...")
        all_chains, best_idx = _run_nuts_chains(fn, base_params, warmup, samples, fit_replicates)
        best_samples = all_chains[best_idx]
        mean_params = jax.tree.map(lambda x: jnp.mean(x, axis=0), best_samples)

        if fit_replicates > 1:
            plot_chains(all_chains, best_idx)
        if estimate_uncertainty:
            _print_posterior_summary(best_samples)

        is_imputed_np = np.array(is_imputed, dtype=bool) if is_imputed is not None \
                        else np.zeros(len(np.array(positive(mean_params["omega"]))), dtype=bool)
        plot_omega_coloured(mean_params, np.array(is_extracellular), is_imputed_np, np.array(regression_mask, dtype=bool))
        plot_omega_by_domain(mean_params, np.array(is_extracellular), is_imputed_np, np.array(regression_mask, dtype=bool), diversity_mask=mask)
        plt.show()
        return

    if is_extracellular is not None:
        is_extracellular = np.array(is_extracellular)
        regression_mask  = np.array(regression_mask)

        logging.info(f"Running NUTS sampling ({fit_replicates} chain(s), {warmup} warmup / {samples} sample steps)...")
        all_chains, best_idx = _run_nuts_chains(fn, base_params, warmup, samples, fit_replicates)
        best_samples = all_chains[best_idx]
        mean_params = jax.tree.map(lambda x: jnp.mean(x, axis=0), best_samples)

        omega_posterior_mean = np.array(positive(mean_params["omega"]))

        print('Final log-likelihood (at posterior mean): ', fn(mean_params))
        print('GTR posterior means: ', jax.tree.map(positive, jnp.array([
            mean_params["alpha"], mean_params["beta"], mean_params["gamma"],
            mean_params["delta"], mean_params["epsilon"], mean_params["eta"], mean_params["theta"]
        ])))
        print('Omega posterior means: ', omega_posterior_mean)

        if fit_replicates > 1:
            plot_chains(all_chains, best_idx)
        if estimate_uncertainty:
            _print_posterior_summary(best_samples)

        # Posthoc regression on posterior mean omega
        alpha_reg, beta_reg, log_sigma = posthoc_regression(
            omega_posterior_mean, is_extracellular, regression_mask
        )
        print(f'Posthoc regression: alpha_reg={alpha_reg:.4f}, beta_reg={beta_reg:.4f}, sigma={np.exp(log_sigma):.4f}')

        # Build a params-like dict for plotting functions that expect positive(params["omega"])
        params_for_plot = {
            "omega":     softplus_inverse(jnp.array(omega_posterior_mean)),
            "alpha_reg": jnp.array(alpha_reg),
            "beta_reg":  jnp.array(beta_reg),
            "log_sigma": jnp.array(log_sigma),
        }

        is_imputed_np = np.array(is_imputed, dtype=bool) if is_imputed is not None \
                        else np.zeros(len(omega_posterior_mean), dtype=bool)
        reg_mask_np   = np.array(regression_mask, dtype=bool)

        plot_regression(params_for_plot, is_extracellular, is_imputed_np, reg_mask_np)
        plot_omega_by_domain(params_for_plot, is_extracellular, is_imputed_np, reg_mask_np, diversity_mask=mask)
        plt.show()

    else:
        logging.info(f"Running NUTS sampling ({fit_replicates} chain(s), {warmup} warmup / {samples} sample steps)...")
        all_chains, best_idx = _run_nuts_chains(fn, base_params, warmup, samples, fit_replicates)
        best_samples = all_chains[best_idx]
        mean_params = jax.tree.map(lambda x: jnp.mean(x, axis=0), best_samples)

        print('Final log-likelihood (at posterior mean): ', fn(mean_params))
        print('GTR posterior means: ', jax.tree.map(positive, jnp.array([
            mean_params["alpha"], mean_params["beta"], mean_params["gamma"],
            mean_params["delta"], mean_params["epsilon"], mean_params["eta"], mean_params["theta"]
        ])))
        print('Omega posterior means: ', positive(mean_params["omega"]))

        if fit_replicates > 1:
            plot_chains(all_chains, best_idx)
        if estimate_uncertainty:
            _print_posterior_summary(best_samples)

        plt.plot(np.array(positive(mean_params["omega"])), 'o', color='black')
        plt.show()
