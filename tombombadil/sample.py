#!/usr/bin/env python

import csv as _csv
import logging
import numpy as np
import scipy
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import jax.scipy.special as special
from jax.scipy.special import gammaln
import optax
import blackjax
from jax import jit
from jax.flatten_util import ravel_pytree
jax.config.update('jax_enable_x64', True)
import matplotlib.pyplot as plt

from .gtr import build_GTR
from .likelihood import gen_alpha
from .device import platform_summary, validate_platform

SCALAR_PARAM_KEYS = ["alpha", "beta", "gamma", "delta", "epsilon", "theta"]
SCALAR_PARAM_KEYS_WITH_ETA = SCALAR_PARAM_KEYS + ["eta"]
_REGRESSION_KEYS = {"alpha_reg", "beta_reg", "log_sigma"}

@jit
def my_dirichlet_multinomial_logpmf(x, a):
    x = jnp.asarray(x, dtype=jnp.float64)
    a = jnp.asarray(a, dtype=jnp.float64)

    N = jnp.sum(x, axis=-1)
    a0 = jnp.sum(a, axis=-1)

    term1 = gammaln(N + 1) - jnp.sum(gammaln(x + 1), axis=-1)
    term2 = gammaln(a0) - gammaln(N + a0)
    term3 = jnp.sum(gammaln(x + a) - (gammaln(a)), axis=-1)

    #print("term3",term3)
    #print("x",x)
    #print("a",a)
    #jax.debug.print("x = {x}", x=x)
    #jax.debug.print("a = {a}", a=a)
    #jax.debug.print("a = {a}", a=a)
    #test = gammaln(a)
    #jax.debug.print("test = {test}", test=test)
    #jax.debug.print("term1 = {term1}", term1=term1)
    #jax.debug.print("term2 = {term2}", term2=term2)
    #jax.debug.print("term3 = {term3}", term3=term3)

    return term1 + term2 + term3 # gives 1407.2288

# This version is adapted from the scipy implementation
def my_dirichlet_multinomial_logpmf_2(x, a):
    x = jnp.asarray(x)
    a = jnp.asarray(a)

    N = jnp.sum(x, axis=-1)
    a0 = jnp.sum(a, axis=-1)

    out = jnp.asarray(gammaln(a0) + gammaln(N + 1) - gammaln(N + a0))
    out += (gammaln(x + a) - (gammaln(a) + gammaln(x + 1))).sum(axis=-1)

    # The scipy version sets the logpmf to -inf if N and sum(x) disagree, but
    # we're calculating N from x here so not really relevant
    # out = jnp.place(out, N != x.sum(axis=-1), -jnp.inf, inplace=False)

    return out

@jit
def model(alpha, beta, gamma, delta, epsilon, eta, mu, omega, pi_eq, log_pi, pimat, pimatinv, pimult, obs_vec):
    # Calculate substitution rate matrix under neutrality
    #print(pimat)
    #print(pimult)
    #A = build_GTR(alpha, beta, gamma, delta, epsilon, eta, 1, pimat, pimult) # 61x61 subst rate matrix
    #A = build_GTR(1, 1, 1, 1, 1, 1, 1, pimat, pimult) # same as NY98?
    A = build_GTR(alpha, beta, gamma, delta, epsilon, eta, 1, pimat, pimult) # 61x61 subst rate matrix # for building the GTR matrix you want omega=1 (mean mutation rate under neutrality)
    #print(A) # is all zeros at the moment
    #print(pi_eq)
    #print(jnp.diagonal(A))
    #print(-jnp.dot(jnp.diagonal(A), pi_eq))
    meanrate = -jnp.dot(jnp.diagonal(A), pi_eq)
    # Calculate substitution rate matrix
    scale = (mu / 2.0) / meanrate

    A2 = gen_alpha(omega, A, pimat, pimult, pimatinv, scale)
    #alpha = gen_alpha(omega, A, pimat, pimult, pimatinv, scale, alpha, beta, gamma, delta, epsilon, eta) # just for comparing runtime between build_GTR and update_GTR
    #print('alpha: ',alpha)
    #print("obs_vec: ", obs_vec)
    #print("N: ", N)
    #jax.debug.print("alpha = {alpha}", alpha=alpha)
    #jax.debug.print("A = {A}", A=A)
    #jax.debug.print("A2 = {A2}", A2=A2) # these calculations are done twice in one step (jit?) and the second time some NaNs appear in A
    # it seems to come from the parameters but not sure? my analysis in test_fn suggests that the likelihood becomes zero with omega close to zero, no NaNs in parameters needed...?
    #print(np.sum(alpha,axis=1).tolist()) # alpha rows clearly do not sum to one but this is what the pmf is expecting -- a problem? no, for dirichlet not a problem
    #log_prob = scipy.stats.multinomial.pmf(obs_vec, N, alpha) # this is where it breaks but is it because the code is broken or because of lack of diversity? It is not because of the lack of diversity
    #log_prob = scipy.stats.multinomial.logpmf(obs_vec, N, alpha) # this is pmf in John's code but we think it might need to be pmf?
    # log_prob = scipy.stats.dirichlet_multinomial.logpmf(obs_vec, alpha, N) # gives -10.21301 (correct)
    log_prob = my_dirichlet_multinomial_logpmf(obs_vec, A2) # our custom, jnp based dirichlet_multinomial.logpmf but something is wrong in the implementation this function gives us an integer, we want a vector of length 61

    #print("Difference between scipy and custom jax dirichlet-multinomial logpmf:", scipy.stats.dirichlet_multinomial.logpmf(obs_vec, alpha, N) - my_dirichlet_multinomial_logpmf(obs_vec, alpha))
    #print("Difference between scipy and other custom jax dirichlet-multinomial logpmf:", scipy.stats.dirichlet_multinomial.logpmf(obs_vec, alpha, N) - my_dirichlet_multinomial_logpmf_2(obs_vec, alpha))
    
    #print("log_prob_shape",log_prob.shape)
    #print('log_prob: ',log_prob)
    #jax.debug.print("obs_vec = {obs_vec}", obs_vec=obs_vec)
    #jax.debug.print("alpha = {alpha}", alpha=alpha)
    #jax.debug.print("log_prob = {log_prob}", log_prob=log_prob)
    #print('log_prop_pi',log_prob + log_pi)
    #print('logsumexp_prop_pi',special.logsumexp(log_prob + log_pi, axis=0))
    return special.logsumexp(log_prob + log_pi, axis=0) # check that these go in as different arguments

def transforms(X, pi_eq):
    #N = np.sum(X, 0)
    #n_loci = len(N)

    # pi transforms
    log_pi = np.log(pi_eq)
    pimat = np.diag(np.sqrt(pi_eq))
    pimatinv = np.diag(np.divide(1, np.sqrt(pi_eq)))

    pimult = np.zeros((61, 61))
    for j in range(61)  :
        for i in range(61):
            pimult[i, j] = np.sqrt(pi_eq[j] / pi_eq[i])
            #pimult = pimult.at[i,j].set(jnp.sqrt(pi_eq[j] / pi_eq[i]))

    return log_pi, pimat, pimatinv, pimult

def positive(a): # transformation for ensuring positive parameter values in model
        eps = 1e-6
        return jnp.exp(a) + eps
        #return jnp.exp(a)

def softplus_inverse(y, eps=1e-6): # inverse transformation for calculating raw parameter values (e.g. for start values of parameters)
    z = y - eps
    #z = y
    return jnp.log((z))


def natural_to_raw_params(params):
    """Convert positive natural-scale parameters to raw log space."""
    raw = {}
    for key, value in params.items():
        if key == "omega":
            raw[key] = jnp.log(jnp.asarray(value) - 1e-6)
        elif key in _REGRESSION_KEYS:
            raw[key] = jnp.asarray(value, dtype=jnp.float64)
        else:
            raw[key] = jnp.array(softplus_inverse(value), dtype=jnp.float64)
    return raw


def make_mask(X):
    col_max = np.max(X, axis=0)
    col_sum = np.sum(X, axis=0)
    return np.where(col_max == col_sum, 0, 1)


def make_base_params(n_sites, estimate_eta=True):
    """Build default raw initial parameters."""
    params = {
        "alpha": jnp.array(softplus_inverse(1), dtype=jnp.float64),
        "beta": jnp.array(softplus_inverse(1), dtype=jnp.float64),
        "gamma": jnp.array(softplus_inverse(1), dtype=jnp.float64),
        "delta": jnp.array(softplus_inverse(1), dtype=jnp.float64),
        "epsilon": jnp.array(softplus_inverse(1), dtype=jnp.float64),
        "theta": jnp.array(softplus_inverse(0.5), dtype=jnp.float64),
        "omega": jnp.repeat(jnp.array(softplus_inverse(0.5), dtype=jnp.float64), n_sites),
    }
    if estimate_eta:
        params["eta"] = jnp.array(softplus_inverse(1), dtype=jnp.float64)
    return params


def make_param_labels(params):
    return {k: ("vec" if k == "omega" else "scalar") for k in params}
    
def regression_log_likelihood(raw_x, is_extracellular, regression_mask):
    """Hierarchical regression log-likelihood on log(omega).

    Models log(omega_i) = alpha_reg + beta_reg * is_extracellular_i + epsilon_i,
    where epsilon_i ~ Normal(0, sigma^2). alpha_reg and beta_reg are in unconstrained
    space (can be any value). sigma = exp(log_sigma).

    The omega values used are positive(raw_omega), consistent with the main model.
    Sites with regression_mask=0 (NA/unannotated) are excluded from the mean.
    Returns mean per-site log-likelihood over included sites.
    """
    omega = positive(raw_x["omega"])
    log_omega = jnp.log(omega)
    mu = raw_x["alpha_reg"] + raw_x["beta_reg"] * is_extracellular
    sigma = jnp.exp(raw_x["log_sigma"])
    per_site = jax.scipy.stats.norm.logpdf(log_omega, mu, sigma)
    return jnp.sum(per_site * regression_mask) / jnp.sum(regression_mask)


def _log_transform_jacobian(raw_value):
    return raw_value


def prior_log_likelihood(raw_x, n_sites, prior_mode="stan_unconstrained",
                         estimate_eta=True, aggregate="mean"):
    """Log prior contributions for MAP regularisation.

    The data log-likelihood is mean-aggregated over sites (mean(losses)), which
    is (1/n_sites) × Σᵢ log P(Dᵢ | θ). For the MAP to coincide with the mode of
    the true Bayesian posterior Σᵢ log P(Dᵢ | θ) + log P(θ), every prior term
    must enter with the same 1/n_sites weighting.

    ``omega`` is a per-site vector. Its prior is averaged over sites for a
    mean-aggregated likelihood and summed for a sum-aggregated likelihood.
    The GTR rates are global parameters, so their summed prior is divided by
    the number of sites for a mean-aggregated likelihood and left unchanged
    for a sum-aggregated likelihood. ``theta`` is a global scale parameter and
    retains its full prior strength.

    ``prior_mode`` selects the parameterisation convention:
    - ``current``: existing priors without Jacobian terms.
    - ``none``: disable all priors.
    - ``stan_constrained``: constrained-scale priors without Jacobians.
    - ``stan_unconstrained``: constrained priors plus raw-to-natural Jacobians.
    """

    if prior_mode == "none":
        return jnp.array(0.0, dtype=jnp.float64)

    omega = positive(raw_x["omega"])
    if prior_mode == "current":
        omega_terms = jax.scipy.stats.norm.logpdf(jnp.log(omega), jnp.log(0.5), 1.0)
    else:
        omega_terms = (
            jax.scipy.stats.norm.logpdf(jnp.log(omega), jnp.log(0.5), 1.0)
            - jnp.log(omega)
        )
        if prior_mode == "stan_unconstrained":
            omega_terms += _log_transform_jacobian(raw_x["omega"])
    omega_prior = jnp.mean(omega_terms)

    gtr_keys = ["alpha", "beta", "gamma", "delta", "epsilon"]
    if estimate_eta and "eta" in raw_x:
        gtr_keys.append("eta")
    gtr_terms = []
    for k in gtr_keys:
        term = jax.scipy.stats.norm.logpdf(positive(raw_x[k]), 0.0, 1.0)
        if prior_mode == "stan_unconstrained":
            term += _log_transform_jacobian(raw_x[k])
        gtr_terms.append(term)
    gtr_prior = jnp.sum(jnp.array(gtr_terms))

    theta_prior = jax.scipy.stats.norm.logpdf(positive(raw_x["theta"]), 0.0, 1.0)
    if prior_mode == "stan_unconstrained":
        theta_prior += _log_transform_jacobian(raw_x["theta"])

    if aggregate == "sum":
        return jnp.sum(omega_terms) + gtr_prior + theta_prior
    return omega_prior + gtr_prior / n_sites + theta_prior


def make_fn(pi_eq, log_pi, pimat, pimatinv, pimult, X, mask,
            is_extracellular=None, regression_mask=None, regression_weight=0.1,
            include_invariant=True, aggregate="mean",
            prior_mode="stan_unconstrained", estimate_eta=True,
            eigen_jitter=True, omega_floor=True):
    # closure for defining fn (this change is mainly for making the unit testing
    # easier, before it was a closure in run_sampler())
    batched_loss = jax.vmap(
        model,
        in_axes=(None, None, None, None, None, None, None, 0, None, None, None, None, None, 1)
    )
    def f(raw_x):
        x = jax.tree.map(positive, raw_x)

        omega = x["omega"]
        if jnp.ndim(omega) == 0:
            omega = jnp.repeat(omega, X.shape[1])

        if not include_invariant:
            omega = jnp.where(
                mask == 1,
                omega,
                jax.lax.stop_gradient(omega)
            )

        if omega_floor:
            omega = jnp.where(
                omega > 0.01,
                omega,
                jax.lax.stop_gradient(omega)
            )

        eta = x["eta"] if estimate_eta and "eta" in x else jnp.array(1.0, dtype=jnp.float64)
        losses = batched_loss(
            x["alpha"], x["beta"], x["gamma"], x["delta"], x["epsilon"],
            eta, x["theta"], omega, pi_eq, log_pi, pimat, pimatinv, pimult, X
        )
        if include_invariant:
            selected_losses = losses
        else:
            mask_f = mask.astype(jnp.float64)
            selected_losses = losses * mask_f

        if aggregate == "sum":
            total = jnp.sum(selected_losses)
        elif include_invariant:
            total = jnp.mean(selected_losses)
        else:
            mask_f = mask.astype(jnp.float64)
            total = jnp.sum(selected_losses) / jnp.maximum(jnp.sum(mask_f), 1.0)

        total = total + prior_log_likelihood(
            raw_x, X.shape[1], prior_mode=prior_mode,
            estimate_eta=estimate_eta, aggregate=aggregate
        )
        if is_extracellular is not None:
            # regression_weight controls how strongly the regression term influences omega
            # relative to the data likelihood. Values < 1 prevent the regression from
            # overriding strong selection signals (very high or very low omega).
            total = total + regression_weight * regression_log_likelihood(raw_x, is_extracellular, regression_mask)
        return total
    return f


def _optimize_params(fn, params, solver, n_iter, verbose=True, convergence=None):
    """Run the optimization loop and return final params plus convergence metadata."""
    loss_fn = lambda p: -fn(p)
    opt_state = solver.init(params)
    use_convergence = convergence is not None and convergence.get("enabled", False)
    check_every = max(int(convergence.get("check_every", 1)), 1) if use_convergence else 1
    patience = max(int(convergence.get("patience", 1)), 1) if use_convergence else 1
    min_steps = max(int(convergence.get("min_steps", 0)), 0) if use_convergence else 0
    tol = float(convergence.get("tol", 0.0)) if use_convergence else 0.0

    best_params = params
    best_objective = float(fn(params)) if use_convergence else None
    stale_checks = 0
    converged = False
    steps_run = 0

    for step in range(1, n_iter + 1):
        grad = jax.grad(loss_fn)(params)
        updates, opt_state = solver.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        steps_run = step
        if verbose:
            print('parameters: ', jax.tree.map(positive, jnp.array([
                params["alpha"], params["beta"], params["gamma"],
                params["delta"], params["epsilon"], params["theta"]
            ])))
            print('omegas: ', jax.tree.map(positive, params["omega"]))

        if use_convergence and step % check_every == 0:
            current_objective = float(fn(params))
            improvement = current_objective - best_objective
            if current_objective > best_objective:
                best_objective = current_objective
                best_params = params
            if step >= min_steps:
                if improvement <= tol:
                    stale_checks += 1
                else:
                    stale_checks = 0
                if stale_checks >= patience:
                    converged = True
                    break

    if not use_convergence:
        best_params = params
        best_objective = float(fn(params))

    return {
        "params": best_params,
        "n_steps": steps_run,
        "objective": best_objective,
        "converged": converged,
    }


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

    known_ext   = is_ext  & ~is_imp           # annotated extracellular
    known_other = ~is_ext & ~is_imp & reg_m   # annotated non-extracellular
    imputed     = is_imp                       # inferred from neighbours (either direction)
    na_sites    = ~reg_m                       # unresolvable, excluded from regression

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
        f'Per-site ω with domain-informed regression\n'
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


def plot_domain_comparison(omega_baseline, omega_domain, is_extracellular, is_imputed, regression_mask):
    """Lollipop plot comparing per-site omega without and with domain regression.

    Each site is drawn as a vertical stick connecting the baseline estimate (hollow
    circle) to the domain-informed estimate (filled circle). Sites are coloured by
    domain annotation. The omega=1 neutral line is shown prominently.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    omega_bl = np.array(omega_baseline)
    omega_dm = np.array(omega_domain)
    is_ext   = np.array(is_extracellular, dtype=bool)
    is_imp   = np.array(is_imputed,       dtype=bool)
    reg_m    = np.array(regression_mask,  dtype=bool)

    known_ext   = is_ext  & ~is_imp
    known_other = ~is_ext & ~is_imp & reg_m
    imputed     = is_imp
    na_sites    = ~reg_m

    colour_map = {'ext': 'tomato', 'other': 'steelblue', 'imputed': 'grey', 'na': 'lightgrey'}
    sites = np.arange(len(omega_bl))

    fig, ax = plt.subplots(figsize=(14, 5))

    for mask, c in [
        (known_ext,   colour_map['ext']),
        (known_other, colour_map['other']),
        (imputed,     colour_map['imputed']),
        (na_sites,    colour_map['na']),
    ]:
        idx = np.where(mask)[0]
        for i in idx:
            ax.plot([i, i], [omega_bl[i], omega_dm[i]],
                    color=c, alpha=0.45, linewidth=0.7, zorder=2)
        # Baseline: hollow circle
        ax.scatter(sites[mask], omega_bl[mask],
                   color=c, s=12, alpha=0.6, facecolors='none', linewidths=0.7, zorder=3)
        # Domain: filled circle
        ax.scatter(sites[mask], omega_dm[mask],
                   color=c, s=12, alpha=0.85, zorder=4)

    ax.axhline(1.0, color='black', linestyle='-', linewidth=2.5, alpha=0.85,
               label='ω = 1 (neutral)', zorder=5)
    ax.set_yscale('log')
    ax.set_xlabel('Alignment site index')
    ax.set_ylabel('ω (dN/dS, log scale)')
    ax.set_title('Per-site ω: baseline (hollow) vs. domain-informed regression (filled)\n'
                 'Sticks show the shift in estimate per site')

    legend_elements = [
        Patch(facecolor=colour_map['ext'],     label='Extracellular (annotated)'),
        Patch(facecolor=colour_map['other'],   label='Other (annotated)'),
        Patch(facecolor=colour_map['imputed'], label='Imputed'),
        Patch(facecolor=colour_map['na'],      label='Unknown (excluded)'),
        Line2D([0], [0], color='black', linewidth=2.5,              label='ω = 1'),
        Line2D([0], [0], color='black', marker='o', linestyle='None',
               markersize=6, markerfacecolor='none', label='Baseline (no domain)'),
        Line2D([0], [0], color='black', marker='o', linestyle='None',
               markersize=6, label='Domain-informed'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.01, 1),
              borderaxespad=0, fontsize=8)
    plt.tight_layout()
    return fig, ax

def compute_laplace_se(fn, params):
    """Diagonal Laplace approximation: per-parameter standard errors at the MAP.

    Flattens the parameter PyTree to a 1-D vector, computes the full Hessian of
    -fn (the negative log-likelihood), and uses its diagonal to approximate the
    marginal variance of each parameter:

        Var(theta_i) ≈ 1 / H_ii,   H = -d²(log L)/dtheta²

    Standard errors in unconstrained (raw) space and on the natural scale are
    both returned.  For parameters transformed by positive(), the delta method
    gives  se_natural = se_raw * exp(raw),  which equals se_raw * omega for omega
    parameters.  Regression parameters (alpha_reg, beta_reg, log_sigma) are
    already in interpretable unconstrained space; log_sigma is also converted to
    the sigma scale via the same delta method.

    Sites where stop_gradient was applied (no diversity / masked omega) have
    H_ii = 0 and are reported as NaN.

    Args:
        fn:     the log-likelihood function (higher = better)
        params: PyTree of optimised (raw/unconstrained) parameter values

    Returns:
        se_raw:     PyTree matching params, SEs in unconstrained space
        se_natural: PyTree matching params, SEs on natural scale
    """
    flat_params, unflatten = ravel_pytree(params)

    def neg_ll_flat(p):
        return -fn(unflatten(p))

    # Compute only the diagonal of the Hessian via forward-over-reverse AD.
    # For each basis vector eᵢ, jvp(grad, params, eᵢ) returns H @ eᵢ;
    # the i-th element of that product is H_ii. This uses O(n) memory
    # rather than the O(n²) required by the full jax.hessian approach.
    grad_fn = jax.grad(neg_ll_flat)
    n = len(flat_params)
    def hess_diag_i(i):
        e_i = jnp.zeros(n).at[i].set(1.0)
        _, hv = jax.jvp(grad_fn, (flat_params,), (e_i,))
        return hv[i]
    hess_diag = jax.vmap(hess_diag_i)(jnp.arange(n))

    # 1 / H_ii gives the marginal variance under the diagonal approximation.
    # H_ii <= 0 means the likelihood is flat there (masked site) → NaN.
    var_raw = jnp.where(hess_diag > 0, 1.0 / hess_diag, jnp.nan)
    se_raw = unflatten(jnp.sqrt(jnp.clip(var_raw, 0)))

    # Delta method onto natural scale for positive()-transformed parameters.
    # For regression params the raw SE is already interpretable; log_sigma also
    # gets the exp() delta method so we report se_sigma = sigma * se_log_sigma.
    flat_se_raw, _ = ravel_pytree(se_raw)
    se_natural = {}
    for k in params:
        if k in _REGRESSION_KEYS:
            se_natural[k] = se_raw[k]
        else:
            se_natural[k] = se_raw[k] * jnp.exp(params[k])

    return se_raw, se_natural


def _print_laplace_summary(params, se_natural):
    """Print a human-readable summary of MAP estimates ± 1 SE (natural scale)."""
    gtr_keys = ["alpha", "beta", "gamma", "delta", "epsilon", "theta"]
    print("\n--- Laplace approximation (diagonal) ---")
    print("GTR / shared parameters (natural scale):")
    for k in gtr_keys:
        if k in params:
            est = float(positive(params[k]))
            se  = float(se_natural[k])
            print(f"  {k:8s}: {est:.4f} ± {se:.4f}")
    if "alpha_reg" in params:
        print("Regression parameters:")
        print(f"  alpha_reg: {float(params['alpha_reg']):.4f} ± {float(se_natural['alpha_reg']):.4f}  (log-omega scale)")
        print(f"  beta_reg:  {float(params['beta_reg']):.4f}  ± {float(se_natural['beta_reg']):.4f}  (log-omega scale)")
        sigma = float(jnp.exp(params["log_sigma"]))
        se_s  = float(se_natural["log_sigma"]) * sigma  # delta method on sigma
        print(f"  sigma:     {sigma:.4f} ± {se_s:.4f}")
    omega_se = np.array(se_natural["omega"])
    valid = omega_se[~np.isnan(omega_se)]
    n_nan = int(np.isnan(omega_se).sum())
    print(f"Omega SEs (natural scale): mean={valid.mean():.4f}  min={valid.min():.4f}  "
          f"max={valid.max():.4f}  ({n_nan} sites masked/NaN)")
    print("----------------------------------------\n")


def plot_omega(sites, omega, variant=None, log_scale=True, true_omega=None):
    """Create a per-site omega scatter plot for saved parameter estimates."""
    sites = np.asarray(sites)
    omega = np.asarray(omega)
    fig, ax = plt.subplots(figsize=(16, 4))

    if variant is None:
        ax.scatter(sites, omega, color="black", s=8, alpha=0.65,
                   linewidths=0, label="Omega estimate", zorder=3)
    else:
        variant = np.asarray(variant)
        if variant.shape != omega.shape:
            raise ValueError("Variant mask length does not match omega length")
        invariant = variant == 0
        variable = ~invariant
        if invariant.any():
            ax.scatter(sites[invariant], omega[invariant], color="lightgrey",
                       s=7, alpha=0.7, linewidths=0, label="Invariant site",
                       zorder=2)
        if variable.any():
            ax.scatter(sites[variable], omega[variable], color="black",
                       s=9, alpha=0.75, linewidths=0, label="Variable site",
                       zorder=3)

    if true_omega is not None and not log_scale:
        true_omega = np.asarray(true_omega)
        if true_omega.shape != omega.shape:
            raise ValueError("True omega length does not match omega length")
        valid_truth = np.isfinite(true_omega)
        ax.vlines(sites[valid_truth], omega[valid_truth], true_omega[valid_truth],
                  color="blue", linewidth=0.6,
                  alpha=0.5, zorder=4)
        ax.scatter(sites[valid_truth], true_omega[valid_truth], color="blue",
                   s=8, alpha=0.7,
                   linewidths=0, label="True omega", zorder=5)

    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.1, alpha=0.8,
               label="omega = 1", zorder=4)
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Alignment site")
    ax.set_ylabel("omega (dN/dS, log scale)" if log_scale else "omega (dN/dS)")
    ax.set_title("Per-site omega estimates" + ("" if log_scale else " (natural scale)"))
    ax.set_xlim(sites.min() - 1, sites.max() + 1)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0,
              fontsize=8)
    fig.tight_layout()
    return fig


def save_params(output_stem: str, params: dict, mask: np.ndarray, estimate_eta=True,
                benchmark_omega=None) -> None:
    """Save MAP parameter estimates to two CSV files.

    {output_stem}_omega.csv   — per-site omega (site, omega_map, variant)
    {output_stem}_scalar.csv  — scalar parameters (variable, value)

    GTR parameters (alpha … theta) are on the natural scale. Regression
    parameters alpha_reg and beta_reg are on the log-omega scale (their
    natural parameterisation); sigma is exp(log_sigma).
    """
    omega_path  = output_stem + "_omega.csv"
    scalar_path = output_stem + "_scalar.csv"

    omega = np.array(positive(params["omega"]))
    if omega.ndim == 0:
        omega = np.repeat(omega, len(mask))
    with open(omega_path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["site", "omega_map", "variant"])
        for i, (om, mk) in enumerate(zip(omega, mask), start=1):
            w.writerow([i, float(om), int(mk)])
    logging.info("Saved omega estimates to: %s", omega_path)

    omega_plot_path = output_stem + "_omega_plot.pdf"
    omega_fig = plot_omega(np.arange(1, len(omega) + 1), omega, mask, log_scale=True)
    omega_fig.savefig(omega_plot_path, format="pdf", bbox_inches="tight")
    plt.close(omega_fig)
    logging.info("Saved omega plot to: %s", omega_plot_path)

    omega_natural_plot_path = output_stem + "_omega_plot_natural.pdf"
    omega_natural_fig = plot_omega(
        np.arange(1, len(omega) + 1), omega, mask, log_scale=False,
        true_omega=benchmark_omega
    )
    omega_natural_fig.savefig(
        omega_natural_plot_path, format="pdf", bbox_inches="tight"
    )
    plt.close(omega_natural_fig)
    logging.info("Saved natural-scale omega plot to: %s", omega_natural_plot_path)

    scalar_keys = ["alpha", "beta", "gamma", "delta", "epsilon", "theta"]
    if estimate_eta:
        scalar_keys = ["alpha", "beta", "gamma", "delta", "epsilon", "eta", "theta"]
    rows = [(k, float(positive(params[k]))) for k in scalar_keys if k in params]
    if "alpha_reg" in params:
        rows += [
            ("alpha_reg", float(params["alpha_reg"])),
            ("beta_reg",  float(params["beta_reg"])),
            ("sigma",     float(jnp.exp(params["log_sigma"]))),
        ]
    with open(scalar_path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["variable", "value"])
        w.writerows(rows)
    logging.info("Saved scalar parameters to: %s", scalar_path)


def _posterior_draws_natural(raw_samples):
    natural = {}
    for k, v in raw_samples.items():
        if k in _REGRESSION_KEYS:
            natural[k] = v
        else:
            natural[k] = positive(v)
    return natural


def summarize_posterior_samples(raw_samples, infos):
    """Summarize posterior samples and BlackJAX diagnostics on natural scale."""
    samples = _posterior_draws_natural(raw_samples)
    summaries = {}
    for k, vals in samples.items():
        arr = np.asarray(vals)
        flat = arr.reshape(-1)
        summaries[k] = {
            "mean": float(np.mean(flat)),
            "sd": float(np.std(flat, ddof=1)) if flat.size > 1 else 0.0,
            "median": float(np.quantile(flat, 0.5)),
            "q2.5": float(np.quantile(flat, 0.025)),
            "q25": float(np.quantile(flat, 0.25)),
            "q75": float(np.quantile(flat, 0.75)),
            "q97.5": float(np.quantile(flat, 0.975)),
            "ess": float(np.mean(blackjax.ess(vals))),
            "rhat": float(np.mean(blackjax.rhat(vals))) if arr.shape[0] > 1 else np.nan,
        }

    diagnostics = {
        "mean_acceptance_rate": float(jnp.mean(infos["acceptance_rate"])),
        "n_divergent": int(jnp.sum(infos["is_divergent"])),
    }
    return samples, summaries, diagnostics


def save_posterior_outputs(output_stem, raw_samples, summaries):
    """Save posterior draws and scalar summaries to CSV files."""
    samples = _posterior_draws_natural(raw_samples)
    keys = list(samples.keys())
    samples_path = output_stem + "_posterior_samples.csv"
    with open(samples_path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["chain", "draw"] + keys)
        n_chains, n_draws = np.asarray(samples[keys[0]]).shape[:2]
        for chain in range(n_chains):
            for draw in range(n_draws):
                row = [chain, draw]
                for k in keys:
                    value = np.asarray(samples[k][chain, draw])
                    row.append(value.tolist() if value.ndim else float(value))
                w.writerow(row)

    summary_path = output_stem + "_posterior_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = _csv.writer(f)
        fields = ["variable", "mean", "sd", "median", "q2.5", "q25", "q75", "q97.5", "ess", "rhat"]
        w.writerow(fields)
        for k in keys:
            row = summaries[k]
            w.writerow([k] + [row[field] for field in fields[1:]])

    logging.info("Saved posterior samples to: %s", samples_path)
    logging.info("Saved posterior summary to: %s", summary_path)


def _save_nuts_point_estimates(output_stem, samples, mask, benchmark_omega=None):
    """Write posterior mean estimates in the same CSV shape as MAP output."""
    mean_params = {}
    for key, value in samples.items():
        mean = jnp.mean(value, axis=(0, 1))
        if key in _REGRESSION_KEYS:
            mean_params[key] = mean
        else:
            mean_params[key] = jnp.log(jnp.maximum(mean - 1e-6, 1e-12))
    save_params(output_stem, mean_params, mask,
                benchmark_omega=benchmark_omega)


def _sample_blackjax_chain(logdensity_fn, initial_position, rng_key, num_warmup,
                           num_samples, target_acceptance_rate):
    """Warm up and sample one NUTS chain with BlackJAX."""
    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        logdensity_fn,
        target_acceptance_rate=target_acceptance_rate,
    )
    warmup_key, sample_key = jax.random.split(rng_key)
    (state, parameters), _ = warmup.run(warmup_key, initial_position, num_steps=num_warmup)
    kernel = blackjax.nuts(logdensity_fn, **parameters).step

    @jax.jit
    def one_step(current_state, step_key):
        new_state, info = kernel(step_key, current_state)
        sample_info = {
            "acceptance_rate": info.acceptance_rate,
            "is_divergent": info.is_divergent,
        }
        return new_state, (new_state.position, sample_info)

    sample_keys = jax.random.split(sample_key, num_samples)
    _, (positions, infos) = jax.lax.scan(one_step, state, sample_keys)
    return positions, infos, parameters


def _sample_blackjax_chains_pmap(logdensity_fn, initial_positions, rng_keys,
                                 num_warmup, num_samples,
                                 target_acceptance_rate):
    """Warm up and sample NUTS chains in parallel across JAX devices."""
    num_chains = int(rng_keys.shape[0])
    n_devices = jax.local_device_count()
    if num_chains > n_devices:
        raise ValueError(
            f"Requested {num_chains} pmap NUTS chain(s), but JAX sees only "
            f"{n_devices} local device(s). Select sequential chains or make "
            "more devices visible to JAX."
        )

    def run_chain(initial_position, rng_key):
        return _sample_blackjax_chain(
            logdensity_fn,
            initial_position,
            rng_key,
            num_warmup,
            num_samples,
            target_acceptance_rate,
        )

    return jax.pmap(run_chain)(initial_positions, rng_keys)


def _stack_chain_pytrees(chain_pytrees):
    return jax.tree.map(lambda *xs: jnp.stack(xs), *chain_pytrees)


def run_nuts_sampler(fn, start_params, num_warmup=1000, num_samples=1000,
                     num_chains=4, rng_seed=0, target_acceptance_rate=0.8,
                     output=None, print_summary=True,
                     chain_mode="sequential", mask=None, benchmark_omega=None):
    """Run BlackJAX NUTS from raw unconstrained starting parameters."""
    if chain_mode not in ("sequential", "pmap"):
        raise ValueError(f"Unknown NUTS chain mode: {chain_mode}")

    rng_key = jax.random.PRNGKey(rng_seed)
    chain_keys = jax.random.split(rng_key, num_chains)
    initial_positions = []
    for chain in range(num_chains):
        if chain == 0:
            initial_positions.append(start_params)
        else:
            initial_positions.append(_perturb_params(start_params))

    if chain_mode == "pmap":
        logging.info(
            "Running %s BlackJAX NUTS chain(s) with pmap across %s local JAX device(s)",
            num_chains, jax.local_device_count(),
        )
        stacked_initial_positions = _stack_chain_pytrees(initial_positions)
        raw_samples, infos, adapted_parameters = _sample_blackjax_chains_pmap(
            fn,
            stacked_initial_positions,
            chain_keys,
            num_warmup,
            num_samples,
            target_acceptance_rate,
        )
        adapted_parameters = [adapted_parameters]
    else:
        chain_positions = []
        chain_infos = []
        adapted_parameters = []
        for chain in range(num_chains):
            logging.info("Running BlackJAX NUTS chain %s/%s", chain + 1, num_chains)
            positions, infos, parameters = _sample_blackjax_chain(
                fn,
                initial_positions[chain],
                chain_keys[chain],
                num_warmup,
                num_samples,
                target_acceptance_rate,
            )
            chain_positions.append(positions)
            chain_infos.append(infos)
            adapted_parameters.append(parameters)

        raw_samples = _stack_chain_pytrees(chain_positions)
        infos = _stack_chain_pytrees(chain_infos)

    natural_samples, summaries, diagnostics = summarize_posterior_samples(raw_samples, infos)
    if print_summary:
        print("\n--- BlackJAX NUTS posterior summary (natural scale) ---")
        for k, s in summaries.items():
            print(
                f"  {k:12s}: mean={s['mean']:.4f}, median={s['median']:.4f}, "
                f"95% CI=({s['q2.5']:.4f}, {s['q97.5']:.4f}), "
                f"ESS={s['ess']:.1f}, R-hat={s['rhat']:.4f}"
            )
        print(
            f"Diagnostics: mean acceptance={diagnostics['mean_acceptance_rate']:.4f}, "
            f"divergences={diagnostics['n_divergent']}"
        )
        print("------------------------------------------------------\n")
    if output is not None:
        save_posterior_outputs(output, raw_samples, summaries)
        if mask is not None:
            _save_nuts_point_estimates(
                output, natural_samples, mask, benchmark_omega=benchmark_omega
            )

    return {
        "raw_samples": raw_samples,
        "samples": natural_samples,
        "summaries": summaries,
        "diagnostics": diagnostics,
        "infos": infos,
        "adapted_parameters": adapted_parameters,
    }


def _perturb_params(params, scale=0.5):
    """Add Normal(0, scale) noise to all parameters in raw (unconstrained) space."""
    key = jax.random.PRNGKey(int(np.random.randint(0, 2**31)))
    flat, unflatten = ravel_pytree(params)
    noise = jax.random.normal(key, flat.shape) * scale
    return unflatten(flat + noise)


def _run_replicates(fn, start_params, param_labels, n_reps, n_iter=100, convergence=None):
    """Run the optimizer n_reps times and return all results plus the index of the best.

    Replicate 0 uses the unperturbed starting point; subsequent replicates add
    Normal(0, 0.5) noise in raw parameter space. All runs are silent (verbose=False).

    Args:
        fn:            log-likelihood function
        start_params:  unperturbed starting parameter dict
        param_labels:  optax multi_transform label dict
        n_reps:        number of restarts
        n_iter:        optimisation iterations per replicate

    Returns:
        all_params: list of param dicts, one per replicate
        best_idx:   index of the replicate with the highest log-likelihood
    """
    all_params = []
    all_lls = []
    for rep in range(n_reps):
        start = dict(start_params) if rep == 0 else _perturb_params(start_params)
        schedule = optax.cosine_decay_schedule(
            init_value=0.2, decay_steps=n_iter, alpha=1e-3 / 0.2
        )
        solver = optax.multi_transform(
            {"vec": optax.adam(schedule), "scalar": optax.adam(schedule)},
            param_labels=param_labels,
        )
        result = _optimize_params(fn, start, solver, n_iter, verbose=False, convergence=convergence)
        params_rep = result["params"]
        ll = result["objective"]
        all_params.append(params_rep)
        all_lls.append(ll)
        status = "converged" if result["converged"] else "max steps"
        logging.info(
            f"  Replicate {rep + 1}/{n_reps}: log-likelihood = {ll:.4f}; "
            f"steps = {result['n_steps']}; status = {status}"
        )
    best_idx = int(np.argmax(all_lls))
    logging.info(f"Best replicate: {best_idx + 1} (log-likelihood = {all_lls[best_idx]:.4f})")
    return all_params, best_idx


def plot_replicates(all_params_list, best_idx):
    """Plot per-site omega and GTR parameter estimates across replicate runs.

    Two panels:
      - Top: omega scatter per site, one colour per replicate; best replicate
             is opaque and larger, others are semi-transparent.
      - Bottom: GTR scalar parameter estimates as a strip plot across replicates.

    Tight clustering indicates robust convergence; spread indicates multiple
    local optima or insufficient iterations.
    """
    n_reps = len(all_params_list)
    all_omegas = [np.array(positive(p["omega"])) for p in all_params_list]
    n_sites = len(all_omegas[0])
    sites = np.arange(n_sites)
    gtr_keys = ["alpha", "beta", "gamma", "delta", "epsilon", "theta"]
    cmap = plt.cm.tab10

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Top: omega per site
    ax = axes[0]
    for i, omega in enumerate(all_omegas):
        is_best = (i == best_idx)
        ax.scatter(sites, omega, color=cmap(i % 10),
                   alpha=0.9 if is_best else 0.25,
                   s=15 if is_best else 8,
                   zorder=4 if is_best else 2,
                   label=f'Rep {i + 1} (best)' if is_best else f'Rep {i + 1}')
    ax.axhline(1.0, color='black', linestyle='-', linewidth=2.5, alpha=0.85,
               label='ω = 1 (neutral)', zorder=5)
    ax.set_yscale('log')
    ax.set_xlabel('Alignment site index')
    ax.set_ylabel('ω (log scale)')
    ax.set_title(f'Per-site ω across {n_reps} replicates — best replicate highlighted')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=8)

    # Bottom: GTR scalar params
    ax2 = axes[1]
    x_pos = np.arange(len(gtr_keys))
    for i, params in enumerate(all_params_list):
        is_best = (i == best_idx)
        vals = [float(positive(params[k])) for k in gtr_keys if k in params]
        ax2.scatter(x_pos[:len(vals)], vals, color=cmap(i % 10),
                    alpha=0.9 if is_best else 0.35,
                    s=60 if is_best else 25,
                    zorder=4 if is_best else 2)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(gtr_keys)
    ax2.axhline(1.0, color='black', linestyle='--', linewidth=1.0, alpha=0.5)
    ax2.set_ylabel('Estimate (natural scale)')
    ax2.set_title('GTR parameter estimates across replicates')

    plt.tight_layout()
    return fig, axes


def plot_omega_coloured(params, is_extracellular, is_imputed, regression_mask):
    """Plot per-site omega estimates coloured by domain type, without regression lines.

    Intended for use with --only-colour-domains: shows domain-coloured omega from a
    standard (no regression) run.
    """
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

    # Shared log-spaced bins across both groups
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
        # Outline for clarity
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


def run_sampler(X, pi_eq, samples=500, platform='cpu', threads=8,
                is_extracellular=None, is_imputed=None, regression_mask=None,
                regression_weight=0.1, only_colour_domains=False,
                estimate_uncertainty=False, fit_replicates=1,
                include_invariant=True, output=None, aggregate="mean",
                benchmark_omega=None,
                prior_mode="stan_unconstrained", estimate_eta=True,
                eigen_jitter=True,
                omega_floor=True, fit_until_convergence=False,
                convergence_tol=1e-6, convergence_patience=5,
                convergence_check_every=10, convergence_min_steps=50,
                fit_method="map", num_warmup=1000, num_samples=1000,
                num_chains=4, rng_seed=0, target_acceptance_rate=0.8,
                nuts_chain_mode="sequential"):
    devices, _ = validate_platform(platform)
    logging.info("JAX execution: %s", platform_summary(platform))
    if nuts_chain_mode == "pmap" and num_chains > len(devices):
        raise ValueError(
            f"Requested {num_chains} pmap NUTS chain(s), but platform '{platform}' "
            f"has only {len(devices)} visible device(s)."
        )
    logging.info("Precomputing transforms...")
    log_pi, pimat, pimatinv, pimult = transforms(X, pi_eq)
    mask = make_mask(X)
    logging.info("Compiling model...")

    base_params = make_base_params(X.shape[1], estimate_eta=estimate_eta)
    base_labels = make_param_labels(base_params)

    if is_extracellular is not None:
        is_extracellular = jnp.array(is_extracellular, dtype=jnp.float64)
        regression_mask = jnp.array(regression_mask, dtype=jnp.float64)

    def make_standard_fn():
        return make_fn(
            pi_eq, log_pi, pimat, pimatinv, pimult, X, mask,
            include_invariant=include_invariant,
            aggregate=aggregate,
            prior_mode=prior_mode,
            estimate_eta=estimate_eta,
            eigen_jitter=eigen_jitter,
            omega_floor=omega_floor,
        )

    def make_domain_fn():
        return make_fn(
            pi_eq, log_pi, pimat, pimatinv, pimult, X, mask,
            is_extracellular, regression_mask, regression_weight,
            include_invariant=include_invariant,
            aggregate=aggregate,
            prior_mode=prior_mode,
            estimate_eta=estimate_eta,
            eigen_jitter=eigen_jitter,
            omega_floor=omega_floor,
        )

    if fit_method == "nuts":
        fn = make_domain_fn() if is_extracellular is not None else make_standard_fn()
        if only_colour_domains:
            logging.warning("--only-colour-domains is ignored with --fit-method nuts")
        logging.info(
            "Running BlackJAX NUTS — %s chain(s), %s warmup step(s), %s draw(s)",
            num_chains, num_warmup, num_samples,
        )
        return run_nuts_sampler(
            fn,
            base_params,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            rng_seed=rng_seed,
            target_acceptance_rate=target_acceptance_rate,
            output=output,
            chain_mode=nuts_chain_mode,
            mask=mask,
            benchmark_omega=benchmark_omega,
        )

    convergence = None
    if fit_until_convergence:
        convergence = {
            "enabled": True,
            "tol": convergence_tol,
            "patience": convergence_patience,
            "check_every": convergence_check_every,
            "min_steps": convergence_min_steps,
        }

    if only_colour_domains and is_extracellular is not None:
        logging.info(
            "Running optimization (no regression, domain colours only) — %s replicate(s)...",
            fit_replicates,
        )
        fn = make_standard_fn()
        all_params, best_idx = _run_replicates(
            fn, base_params, base_labels, fit_replicates, n_iter=samples, convergence=convergence
        )
        params = all_params[best_idx]
        if fit_replicates > 1:
            plot_replicates(all_params, best_idx)
        if estimate_uncertainty:
            logging.info("Computing Laplace uncertainty...")
            _, se_nat = compute_laplace_se(fn, params)
            _print_laplace_summary(params, se_nat)
        is_imputed_np = np.array(is_imputed, dtype=bool) if is_imputed is not None else np.zeros(len(positive(params["omega"])), dtype=bool)
        plot_omega_coloured(params, np.array(is_extracellular), is_imputed_np, np.array(regression_mask, dtype=bool))
        plot_omega_by_domain(params, np.array(is_extracellular), is_imputed_np, np.array(regression_mask, dtype=bool), diversity_mask=mask)
        if output is not None:
            save_params(output, params, mask,
                        benchmark_omega=benchmark_omega)
        plt.show()
        return

    if is_extracellular is not None:
        logging.info("Running baseline optimization (no domain regression)...")
        fn_baseline = make_standard_fn()
        baseline_all, baseline_best = _run_replicates(
            fn_baseline, base_params, base_labels, 1, n_iter=samples, convergence=convergence
        )
        params_baseline = baseline_all[baseline_best]
        omega_baseline = positive(params_baseline["omega"])
        if estimate_uncertainty:
            logging.info("Computing baseline Laplace uncertainty...")
            _, se_nat_baseline = compute_laplace_se(fn_baseline, params_baseline)
            _print_laplace_summary(params_baseline, se_nat_baseline)

        logging.info("Running optimization with domain regression — %s replicate(s)...", fit_replicates)
        domain_params = dict(base_params)
        domain_params["alpha_reg"] = jnp.array(0.0, dtype=jnp.float64)
        domain_params["beta_reg"] = jnp.array(0.0, dtype=jnp.float64)
        domain_params["log_sigma"] = jnp.array(jnp.log(2.0), dtype=jnp.float64)
        domain_labels = dict(base_labels)
        domain_labels["alpha_reg"] = "scalar"
        domain_labels["beta_reg"] = "scalar"
        domain_labels["log_sigma"] = "scalar"

        fn_domain = make_domain_fn()
        all_domain_params, best_idx = _run_replicates(
            fn_domain, domain_params, domain_labels, fit_replicates, n_iter=samples, convergence=convergence
        )
        params = all_domain_params[best_idx]
        omega_domain = positive(params["omega"])

        loss_fn = lambda p: -fn_domain(p)
        print('Final likelihood: ', fn_domain(params))
        print('final parameters: ', jax.tree.map(positive, jnp.array([
            params["alpha"], params["beta"], params["gamma"],
            params["delta"], params["epsilon"], params["theta"]
        ])))
        print('final omega: ', omega_domain)
        print('final alpha_reg: ', params["alpha_reg"])
        print('final beta_reg: ', params["beta_reg"])
        print('final sigma: ', jnp.exp(params["log_sigma"]))
        print('Objective function: ', loss_fn(params))
        if fit_replicates > 1:
            plot_replicates(all_domain_params, best_idx)
        if estimate_uncertainty:
            logging.info("Computing domain-run Laplace uncertainty...")
            _, se_nat_domain = compute_laplace_se(fn_domain, params)
            _print_laplace_summary(params, se_nat_domain)

        is_imputed_np = np.array(is_imputed, dtype=bool) if is_imputed is not None else np.zeros(len(omega_domain), dtype=bool)
        is_ext_np = np.array(is_extracellular)
        reg_mask_np = np.array(regression_mask, dtype=bool)

        plot_regression(params, is_ext_np, is_imputed_np, reg_mask_np)
        plot_domain_comparison(omega_baseline, omega_domain, is_ext_np, is_imputed_np, reg_mask_np)
        plot_omega_by_domain(params, is_ext_np, is_imputed_np, reg_mask_np, diversity_mask=mask)
        if output is not None:
            save_params(output, params, mask,
                        benchmark_omega=benchmark_omega)
        plt.show()
        return

    logging.info("Running optimization — %s replicate(s)...", fit_replicates)
    fn = make_standard_fn()
    all_params, best_idx = _run_replicates(
        fn, base_params, base_labels, fit_replicates, n_iter=samples, convergence=convergence
    )
    params = all_params[best_idx]

    loss_fn = lambda p: -fn(p)
    print('Final likelihood: ', fn(params))
    if estimate_eta:
        print('final parameters: ', jax.tree.map(positive, jnp.array([
                params["alpha"], params["beta"], params["gamma"],
                params["delta"], params["epsilon"],params["eta"], params["theta"]
                ])))
        print('transition/transversion ratio: ', 
                ((jax.tree.map(positive,params["alpha"]) +jax.tree.map(positive,params["eta"]))/(jax.tree.map(positive,params["beta"]) + jax.tree.map(positive,params["gamma"]) + jax.tree.map(positive,params["delta"]) + jax.tree.map(positive,params["epsilon"])))
            )
    else:
        print('final parameters: ', jax.tree.map(positive, jnp.array([
            params["alpha"], params["beta"], params["gamma"],
            params["delta"], params["epsilon"], params["theta"]
        ])))
        print('transition/transversion ratio: ', 
                    ((jax.tree.map(positive,params["alpha"]) +1)/(jax.tree.map(positive,params["beta"]) + jax.tree.map(positive,params["gamma"]) + jax.tree.map(positive,params["delta"]) + jax.tree.map(positive,params["epsilon"])))
                )
    
    print('final omega: ', jax.tree.map(positive, params["omega"]))
    print('Objective function: ', loss_fn(params))
    if fit_replicates > 1:
        plot_replicates(all_params, best_idx)
    if estimate_uncertainty:
        logging.info("Computing Laplace uncertainty...")
        _, se_nat = compute_laplace_se(fn, params)
        _print_laplace_summary(params, se_nat)

    if output is not None:
        save_params(output, params, mask, estimate_eta,
                    benchmark_omega=benchmark_omega)
