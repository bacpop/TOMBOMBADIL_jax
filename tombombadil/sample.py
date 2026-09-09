#!/usr/bin/env python

import csv as _csv
import logging
import os
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.special as special
from jax.scipy.special import gammaln
import blackjax
import optax
from jax import jit
from jax.flatten_util import ravel_pytree
jax.config.update('jax_enable_x64', True)
import matplotlib.pyplot as plt

from .gtr import build_GTR
from .likelihood import gen_alpha
from .likelihood import gen_alpha_no_jitter


SCALAR_PARAM_KEYS_WITH_ETA = ["alpha", "beta", "gamma", "delta", "epsilon", "eta", "theta", "omega"]
GTR_PARAM_KEYS_WITH_ETA = ["alpha", "beta", "gamma", "delta", "epsilon", "eta", "theta"]
OMEGA_MODES = ("scalar", "per-site")


def validate_omega_mode(omega_mode):
    if omega_mode not in OMEGA_MODES:
        raise ValueError(f"Unknown omega mode: {omega_mode!r}")
    return omega_mode


def mode_output_stem(output_stem, omega_mode):
    """Prefix an output stem so files identify their omega parameterisation."""
    validate_omega_mode(omega_mode)
    directory, basename = os.path.split(str(output_stem))
    prefixed = f"{omega_mode.replace('-', '_')}_{basename}"
    return os.path.join(directory, prefixed) if directory else prefixed

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


@jit
def model_no_jitter(alpha, beta, gamma, delta, epsilon, eta, mu, omega, pi_eq, log_pi, pimat, pimatinv, pimult, obs_vec):
    A = build_GTR(alpha, beta, gamma, delta, epsilon, eta, 1, pimat, pimult)
    meanrate = -jnp.dot(jnp.diagonal(A), pi_eq)
    scale = (mu / 2.0) / meanrate
    A2 = gen_alpha_no_jitter(omega, A, pimat, pimult, pimatinv, scale)
    log_prob = my_dirichlet_multinomial_logpmf(obs_vec, A2)
    return special.logsumexp(log_prob + log_pi, axis=0)

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
    
def _log_transform_jacobian(raw_value):
    return jnp.log(jnp.exp(raw_value))


def prior_log_likelihood(raw_x, n_sites, prior_mode="current", estimate_eta=False,
                         omega_mode="scalar", aggregate="sum"):
    """Log prior contributions for MAP regularisation.

    The data log-likelihood is mean-aggregated over sites (mean(losses)), which
    is (1/n_sites) × Σᵢ log P(Dᵢ | θ). For the MAP to coincide with the mode of
    the true Bayesian posterior Σᵢ log P(Dᵢ | θ) + log P(θ), every prior term
    must enter with the same 1/n_sites weighting.

    omega:     LogNormal(log(0.5), 1). This is now a global scalar, so it is
               weighted like the other global priors.
    GTR rates: Half-normal priors. These are global (one prior per parameter,
               not per site), so we explicitly divide the summed log-prior by
               n_sites to put it at the same effective weight as one per-site
               quantity. eta is fixed to 1.0 (not sampled) to remove the global
               GTR-scale ambiguity, so it is excluded from the prior.
    theta:     Half-normal, same form as the GTR rates but NOT divided by n_sites.
               theta is a global mutation rate scalar that scales the overall branch
               length; its prior is intentionally kept at full strength.
    """

    if prior_mode == "none":
        return jnp.array(0.0, dtype=jnp.float64)

    validate_omega_mode(omega_mode)
    omega = positive(raw_x["omega"])

    if prior_mode == "current":
        omega_prior = jax.scipy.stats.norm.logpdf(jnp.log(omega), jnp.log(0.5), 1.0)
    else:
        omega_prior = (
            jax.scipy.stats.norm.logpdf(jnp.log(omega), jnp.log(0.5), 1.0)
            - jnp.log(omega)
        )
        if prior_mode == "stan_unconstrained":
            omega_prior += _log_transform_jacobian(raw_x["omega"])

    gtr_keys = ["alpha", "beta", "gamma", "delta", "epsilon"]
    if estimate_eta:
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

    if omega_mode == "per-site":
        omega_prior = jnp.sum(omega_prior) if aggregate == "sum" else jnp.mean(omega_prior)

    if prior_mode in ("stan_constrained", "stan_unconstrained"):
        return omega_prior + gtr_prior + theta_prior

    if omega_mode == "per-site":
        return omega_prior + gtr_prior / n_sites + theta_prior
    return (omega_prior + gtr_prior) / n_sites + theta_prior


def make_fn(pi_eq, log_pi, pimat, pimatinv, pimult, X, mask,
            include_invariant=True, aggregate="mean", prior_mode="current",
            estimate_eta=False, eigen_jitter=True, omega_floor=True,
            omega_mode="scalar"): # closure for defining fn
    validate_omega_mode(omega_mode)
    model_fn = model if eigen_jitter else model_no_jitter
    omega_axis = None if omega_mode == "scalar" else 0
    batched_loss = jax.vmap(
        model_fn,
        in_axes=(None, None, None, None, None, None, None, omega_axis,
                 None, None, None, None, None, 1)
    )
    def f(raw_x):

        #x = jnp.exp(x)
        #print('x: ',x)
        #return model(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7:], pi_eq, log_pi, N[col], pimat, pimatinv, pimult, X[:, col])
        #return model(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7:], pi_eq, log_pi, N[col], pimat, pimatinv, pimult, X)
        x = jax.tree.map(positive, raw_x)

        if omega_mode == "per-site" and x["omega"].shape != (X.shape[1],):
            raise ValueError(
                f"Per-site omega must have shape ({X.shape[1]},), got {x['omega'].shape}"
            )

        if omega_floor:
            x["omega"] = jnp.where( # stops gradients for omegas <= 0.01
                x["omega"] > 0.01,
                x["omega"],
                jax.lax.stop_gradient(x["omega"])
            )


        # eta is fixed to 1.0 (no longer sampled) to identify the global GTR rate scale.
        eta = x["eta"] if estimate_eta else jnp.array(1.0, dtype=jnp.float64)
        losses = batched_loss(x["alpha"], x["beta"], x["gamma"], x["delta"], x["epsilon"], eta, x["theta"], x["omega"], pi_eq, log_pi, pimat, pimatinv, pimult, X)
        #print('losses: ',losses)
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
            estimate_eta=estimate_eta, omega_mode=omega_mode, aggregate=aggregate,
        )
        return total
    return f


def natural_to_raw_params(params, omega_mode="scalar"):
    """Convert positive natural-scale parameters to this code's raw log scale."""
    validate_omega_mode(omega_mode)
    return {k: jnp.array(softplus_inverse(v), dtype=jnp.float64) for k, v in params.items()}


def make_mask(X):
    col_max = np.max(X, axis=0)
    col_sum = np.sum(X, axis=0)
    return np.where(col_max == col_sum, 0, 1)


def make_base_params(estimate_eta=True, n_sites=None, omega_mode="scalar"):
    """Build default raw initial parameters for scalar-GTR fitting."""
    validate_omega_mode(omega_mode)
    if omega_mode == "per-site" and n_sites is None:
        raise ValueError("n_sites is required for per-site omega")
    omega = jnp.array(softplus_inverse(0.5), dtype=jnp.float64)
    if omega_mode == "per-site":
        omega = jnp.repeat(omega, int(n_sites))
    params = {
        "alpha":   jnp.array(softplus_inverse(1),   dtype=jnp.float64),
        "beta":    jnp.array(softplus_inverse(1),   dtype=jnp.float64),
        "gamma":   jnp.array(softplus_inverse(1),   dtype=jnp.float64),
        "delta":   jnp.array(softplus_inverse(1),   dtype=jnp.float64),
        "epsilon": jnp.array(softplus_inverse(1),   dtype=jnp.float64),
        "theta":   jnp.array(softplus_inverse(0.5), dtype=jnp.float64),
        "omega":   omega,
    }
    if estimate_eta:
        params["eta"] = jnp.array(softplus_inverse(1), dtype=jnp.float64)
    return params


def make_param_labels(params):
    return {k: ("vec" if k == "omega" and jnp.ndim(v) else "scalar")
            for k, v in params.items()}


def evaluate_fixed_params(X, pi_eq, natural_params, include_invariant=True,
                          aggregate="sum", prior_mode="none",
                          estimate_eta=True, eigen_jitter=False,
                          omega_floor=False, omega_mode="scalar"):
    """Evaluate the scalar-GTR objective at fixed natural-scale parameters."""
    log_pi, pimat, pimatinv, pimult = transforms(X, pi_eq)
    mask = make_mask(X)
    raw_params = natural_to_raw_params(natural_params, omega_mode=omega_mode)
    fn = make_fn(
        pi_eq, log_pi, pimat, pimatinv, pimult, X, mask,
        include_invariant=include_invariant,
        aggregate=aggregate,
        prior_mode=prior_mode,
        estimate_eta=estimate_eta,
        eigen_jitter=eigen_jitter,
        omega_floor=omega_floor,
        omega_mode=omega_mode,
    )
    return float(fn(raw_params))


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


def compute_laplace_se(fn, params):
    """Diagonal Laplace approximation: per-parameter standard errors at the MAP.

    Flattens the parameter PyTree to a 1-D vector, computes the full Hessian of
    -fn (the negative log-likelihood), and uses its diagonal to approximate the
    marginal variance of each parameter:

        Var(theta_i) ≈ 1 / H_ii,   H = -d²(log L)/dtheta²

    Standard errors in unconstrained (raw) space and on the natural scale are
    both returned. For parameters transformed by positive(), the delta method
    gives se_natural = se_raw * exp(raw).

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
    se_natural = {k: se_raw[k] * jnp.exp(params[k]) for k in params}

    return se_raw, se_natural


def _print_laplace_summary(params, se_natural, omega_mode="scalar"):
    """Print a human-readable summary of MAP estimates ± 1 SE (natural scale)."""
    validate_omega_mode(omega_mode)
    gtr_keys = GTR_PARAM_KEYS_WITH_ETA
    print("\n--- Laplace approximation (diagonal) ---")
    print("Scalar parameters (natural scale):")
    for k in gtr_keys:
        if k in params:
            est = float(positive(params[k]))
            se  = float(se_natural[k])
            print(f"  {k:8s}: {est:.4f} ± {se:.4f}")
    if "omega" in params:
        omega = np.asarray(positive(params["omega"]))
        se = np.asarray(se_natural["omega"])
        if omega_mode == "scalar":
            print(f"  {'omega':8s}: {float(omega):.4f} ± {float(se):.4f}")
        else:
            print(f"  {'omega':8s}: {len(omega)} site-wise standard errors")
    print("----------------------------------------\n")


def save_params(output_stem: str, params: dict, mask: np.ndarray = None,
                omega_mode="scalar") -> None:
    """Save MAP scalar parameter estimates to CSV.

    scalar_<stem>_Allparams.csv — scalar-mode parameters
    per_site_<stem>_GTRparams.csv — shared GTR/theta parameters
    """
    validate_omega_mode(omega_mode)
    output_stem = mode_output_stem(output_stem, omega_mode)
    scalar_keys = SCALAR_PARAM_KEYS_WITH_ETA if omega_mode == "scalar" else GTR_PARAM_KEYS_WITH_ETA
    parameter_name = "Allparams" if omega_mode == "scalar" else "GTRparams"
    scalar_path = output_stem + f"_{parameter_name}.csv"
    rows = [(k, float(positive(params[k]))) for k in scalar_keys if k in params]
    with open(scalar_path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["variable", "value"])
        w.writerows(rows)
    logging.info("Saved scalar parameters to: %s", scalar_path)
    if omega_mode == "per-site":
        if mask is None:
            mask = np.ones(len(np.asarray(params["omega"])), dtype=int)
        omega_path = output_stem + "_omega.csv"
        with open(omega_path, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["site", "omega_map", "variant"])
            for i, (value, variant) in enumerate(zip(np.asarray(positive(params["omega"])), mask), start=1):
                w.writerow([i, float(value), int(variant)])
        logging.info("Saved per-site omega estimates to: %s", omega_path)


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
            f"{n_devices} local device(s). On CPU, run through the CLI with "
            f"--nuts-chain-mode pmap --cpus {num_chains} before JAX is imported, "
            "or use --nuts-chain-mode sequential."
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


def _posterior_draws_natural(raw_samples):
    return {k: positive(v) for k, v in raw_samples.items()}


def summarize_posterior_samples(raw_samples, infos, omega_mode="scalar"):
    """Summarize posterior samples and BlackJAX diagnostics on natural scale."""
    samples = _posterior_draws_natural(raw_samples)
    validate_omega_mode(omega_mode)
    summaries = {}
    for k in SCALAR_PARAM_KEYS_WITH_ETA:
        if k not in samples:
            continue
        vals = np.asarray(samples[k])
        flat = vals.reshape(-1) if omega_mode == "scalar" or k != "omega" else vals.reshape(-1, vals.shape[-1])
        if omega_mode == "per-site" and k == "omega":
            summaries[k] = {
                "mean": np.mean(flat, axis=0), "sd": np.std(flat, axis=0, ddof=1),
                "median": np.quantile(flat, 0.5, axis=0),
                "q2.5": np.quantile(flat, 0.025, axis=0), "q25": np.quantile(flat, 0.25, axis=0),
                "q75": np.quantile(flat, 0.75, axis=0), "q97.5": np.quantile(flat, 0.975, axis=0),
                "ess": np.asarray(blackjax.ess(samples[k])),
                "rhat": np.asarray(blackjax.rhat(samples[k])) if vals.shape[0] > 1 else np.full(vals.shape[-1], np.nan),
            }
            continue
        summaries[k] = {
            "mean": float(np.mean(flat)),
            "sd": float(np.std(flat, ddof=1)) if flat.size > 1 else 0.0,
            "median": float(np.quantile(flat, 0.5)),
            "q2.5": float(np.quantile(flat, 0.025)),
            "q25": float(np.quantile(flat, 0.25)),
            "q75": float(np.quantile(flat, 0.75)),
            "q97.5": float(np.quantile(flat, 0.975)),
            "ess": float(blackjax.ess(samples[k])),
            "rhat": float(blackjax.rhat(samples[k])) if vals.shape[0] > 1 else np.nan,
        }

    diagnostics = {
        "mean_acceptance_rate": float(jnp.mean(infos["acceptance_rate"])),
        "n_divergent": int(jnp.sum(infos["is_divergent"])),
    }
    return samples, summaries, diagnostics


def save_posterior_outputs(output_stem, raw_samples, summaries, omega_mode="scalar"):
    """Save posterior draws and scalar summaries to CSV files."""
    samples = _posterior_draws_natural(raw_samples)
    validate_omega_mode(omega_mode)
    output_stem = mode_output_stem(output_stem, omega_mode)
    keys = [k for k in SCALAR_PARAM_KEYS_WITH_ETA if k in samples and (omega_mode == "scalar" or k != "omega")]
    samples_path = output_stem + "_posterior_samples.csv"
    with open(samples_path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["chain", "draw"] + keys)
        n_chains, n_draws = np.asarray(samples[keys[0]]).shape[:2]
        for chain in range(n_chains):
            for draw in range(n_draws):
                w.writerow([chain, draw] + [float(samples[k][chain, draw]) for k in keys])

    if omega_mode == "per-site":
        omega_summary_path = output_stem + "_omega_posterior_summary.csv"
        omega = np.asarray(samples["omega"])
        summary = summaries["omega"]
        with open(omega_summary_path, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["site", "mean", "sd", "median", "q2.5", "q25", "q75", "q97.5", "ess", "rhat"])
            for site in range(omega.shape[-1]):
                w.writerow([site + 1] + [float(summary[field][site]) for field in
                                         ("mean", "sd", "median", "q2.5", "q25", "q75", "q97.5", "ess", "rhat")])

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


def _print_posterior_summary(summaries, diagnostics):
    print("\n--- BlackJAX NUTS posterior summary (natural scale) ---")
    for k in SCALAR_PARAM_KEYS_WITH_ETA:
        if k not in summaries:
            continue
        s = summaries[k]
        if np.ndim(s["mean"]) != 0:
            print(f"  {k:8s}: {len(np.asarray(s['mean']))} site-wise posterior summaries")
            continue
        print(
            f"  {k:8s}: mean={s['mean']:.4f}, median={s['median']:.4f}, "
            f"95% CI=({s['q2.5']:.4f}, {s['q97.5']:.4f}), "
            f"ESS={s['ess']:.1f}, R-hat={s['rhat']:.4f}"
        )
    print(
        f"Diagnostics: mean acceptance={diagnostics['mean_acceptance_rate']:.4f}, "
        f"divergences={diagnostics['n_divergent']}"
    )
    print("------------------------------------------------------\n")


def run_nuts_sampler(fn, start_params, num_warmup=1000, num_samples=1000,
                     num_chains=4, rng_seed=0, target_acceptance_rate=0.8,
                     output=None, print_summary=True,
                     chain_mode="sequential", omega_mode="scalar"):
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

    natural_samples, summaries, diagnostics = summarize_posterior_samples(
        raw_samples, infos, omega_mode=omega_mode
    )
    if print_summary:
        _print_posterior_summary(summaries, diagnostics)
    if output is not None:
        save_posterior_outputs(output, raw_samples, summaries, omega_mode=omega_mode)

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
        all_params:   list of param dicts, one per replicate
        best_idx:     index of the replicate with the highest log-likelihood
        all_metadata: optimizer metadata dicts, one per replicate
    """
    all_params = []
    all_lls = []
    all_metadata = []
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
        all_metadata.append(result)
        status = "converged" if result["converged"] else "max steps"
        logging.info(
            f"  Replicate {rep + 1}/{n_reps}: log-likelihood = {ll:.4f}; "
            f"steps = {result['n_steps']}; status = {status}"
        )
    best_idx = int(np.argmax(all_lls))
    logging.info(f"Best replicate: {best_idx + 1} (log-likelihood = {all_lls[best_idx]:.4f})")
    return all_params, best_idx, all_metadata


def plot_replicates(all_params_list, best_idx):
    """Plot scalar parameter estimates across replicate runs."""
    n_reps = len(all_params_list)
    scalar_keys = [k for k in GTR_PARAM_KEYS_WITH_ETA if k in all_params_list[0]]
    cmap = plt.cm.tab10

    fig, ax = plt.subplots(figsize=(10, 4))

    for i, params in enumerate(all_params_list):
        is_best = (i == best_idx)
        vals = [float(positive(params[k])) for k in scalar_keys if k in params]
        ax.scatter(
            np.arange(len(vals)),
            vals,
            color=cmap(i % 10),
            alpha=0.9 if is_best else 0.35,
            s=60 if is_best else 25,
            zorder=4 if is_best else 2,
            label=f'Rep {i + 1} (best)' if is_best else f'Rep {i + 1}',
        )
    ax.set_xticks(np.arange(len(scalar_keys)))
    ax.set_xticklabels(scalar_keys)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1.0, alpha=0.5)
    ax.set_ylabel('Estimate (natural scale)')
    ax.set_title(f'Scalar parameter estimates across {n_reps} replicates')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=8)

    plt.tight_layout()
    return fig, ax


def plot_per_site_omega(params, mask=None, domain_labels=None):
    """Plot per-site omega estimates, optionally coloured by domain labels."""
    omega = np.asarray(positive(params["omega"]))
    sites = np.arange(1, len(omega) + 1)
    fig, ax = plt.subplots(figsize=(12, 4))
    colours = np.full(len(omega), "black", dtype=object)
    if mask is not None:
        colours[np.asarray(mask) == 0] = "lightgrey"
    if domain_labels is not None:
        labels = np.asarray(domain_labels, dtype=object)
        if labels.shape != omega.shape:
            raise ValueError("Domain labels must have one value per alignment site")
        colours = np.where(labels == "extracellular", "tomato", colours)
        colours = np.where(labels == "other", "steelblue", colours)
    ax.scatter(sites, omega, c=colours, s=15, alpha=0.75)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_xlabel("Alignment codon site")
    ax.set_ylabel("omega")
    ax.set_title("Per-site omega estimates")
    fig.tight_layout()
    return fig, ax


def run_sampler(X, pi_eq, samples=500, platform='cpu', threads=8,
                estimate_uncertainty=False, fit_replicates=1,
                include_invariant=True, output=None, aggregate="sum",
                prior_mode="stan_unconstrained", estimate_eta=True,
                eigen_jitter=True, omega_floor=True,
                fit_until_convergence=False, convergence_tol=1e-6,
                convergence_patience=5, convergence_check_every=10,
                convergence_min_steps=50, fit_method="map",
                num_warmup=1000, num_samples=1000, num_chains=4,
                rng_seed=0, target_acceptance_rate=0.8,
                nuts_chain_mode="sequential", omega_mode="scalar",
                domain_labels=None):
    validate_omega_mode(omega_mode)
    logging.info("Precomputing transforms...")
    #col = 30 # site in the alignment
    col = 7 # site in the alignment # this is a column with a bit of diversity (unlike 31)
    # add for loop later
    #X[:,7] = jnp.zeros(61)
    #X[:,7] = np.zeros(61)
    #X[15,7] = 4
    #X[47,7] = 19
    #X[15,7] = 4
    #X[47,7] = 19
    #X = np.zeros((61,1))
    #X[15,:] = 4
    #X[47,:] = 19
    #col = 0
    #X = np.zeros((61,10))
    #X[15,:] = 4
    #X[47,:] = 19
    #col = 0
    #X = np.array(X[:,11:16]) # found some diversity in these columns
    #X = np.array(X[:,11:18]) # found some diversity in these columns, and last column has no diversity
    #X = np.array(X[:,7:18]) # found some diversity in these columns, and last column has no diversity # position 9 is problematic has 5x one codon, 18x another, which corresponds to nonsyn mutation I think (so dS = 0)
    #X = np.array(X[:,9:10])
    #X = np.array(X[:,7:8])
    # probably need exceptions for these cases?
    #print("X shape",X.shape)
    # I think there's a problem with the function reading in the data (the order of the codons)
    """ X = np.zeros((61,5))
    X[9,0] = 5
    X[22,0] = 18
    X[24,1] = 5
    X[37,1] = 17
    X[55,1] = 1
    X[38,2] = 1 # 39  55  60 
    X[54,2] = 5
    X[59,2] = 17
    X[49,3] = 8 # 50  58  59 
    X[57,3] = 13
    X[58,3] = 2
    X[23,4] = 5 # 24  25  40
    X[24,4] = 17
    X[39,4] = 1
    X = np.array(X[:,1:4]) """
    #X = np.array(X[:,10:11]) # this one for example behaves like it has found stop codons, where actually there should be 17x of AAT
    # it should be (based on stan code)
    #X = np.zeros((61,1))
    #X[24,0] = 5
    #X[37,0] = 17
    #X[55,0] = 1
    #X = np.zeros((61,1))
    #X[9,0] = 5
    #X[22,0] = 18
    #X = np.array(X[:,10:14])
    #X = np.array(X[:,0:15])
    log_pi, pimat, pimatinv, pimult = transforms(X, pi_eq)
    # l is length of alignment
    #print("X",X)
    #print("sum X", np.sum(X))
    #print("larger zero", np.where(X > 0))

    # calculate mask for masking parts of the alignment where there is no diversity
    # this will allow using these position for calculating gradient for constant parameters but excludes omega calculation for these positions
    mask = make_mask(X) # create mask for positions without diversity
    #print("col_max",col_max)
    #print("col_sum",col_sum)
    #print("mask",mask)
    #mask = mask.at[0].set(0.0)
    #mask2 = mask ==1
    #print("mask where",mask2)
    #X = X[:,mask2] # this could be an alternative, where I filter X by positions that show diversity
    #print("X",X)
    #log_pi, pimat, pimatinv, pimult = transforms(X, pi_eq)
    logging.info("Compiling model...")

    base_params = make_base_params(
        n_sites=X.shape[1], estimate_eta=estimate_eta, omega_mode=omega_mode
    )
    base_labels = make_param_labels(base_params)
    fn = make_fn(
        pi_eq, log_pi, pimat, pimatinv, pimult, X, mask,
        include_invariant=include_invariant,
        aggregate=aggregate,
        prior_mode=prior_mode,
        estimate_eta=estimate_eta,
        eigen_jitter=eigen_jitter,
        omega_floor=omega_floor,
        omega_mode=omega_mode,
    )

    if fit_method == "nuts":
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
            omega_mode=omega_mode,
        )

    logging.info(f"Running optimization — {fit_replicates} replicate(s)...")
    convergence = None
    if fit_until_convergence:
        convergence = {
            "enabled": True,
            "tol": convergence_tol,
            "patience": convergence_patience,
            "check_every": convergence_check_every,
            "min_steps": convergence_min_steps,
        }
    all_params, best_idx, all_metadata = _run_replicates(
        fn, base_params, base_labels, fit_replicates, n_iter=samples,
        convergence=convergence,
    )
    params = all_params[best_idx]
    best_metadata = all_metadata[best_idx]

    loss_fn = lambda p: -fn(p)
    print('Final likelihood: ', fn(params))
    print('final parameters: ', jax.tree.map(positive, jnp.array([
        params["alpha"], params["beta"], params["gamma"],
        params["delta"], params["epsilon"], params["theta"]
        ])))
    print('final omega: ', jax.tree.map(positive, params["omega"]))
    if estimate_eta:
        print('final eta: ', positive(params["eta"]))
        print('transition/transversion ratio: ', 
            ((jax.tree.map(positive,params["beta"]) +jax.tree.map(positive,params["epsilon"]))/(jax.tree.map(positive,params["alpha"]) + jax.tree.map(positive,params["gamma"]) + jax.tree.map(positive,params["delta"]) + jax.tree.map(positive,params["eta"])))
        )
    print('transition/transversion ratio: ', 
            ((jax.tree.map(positive,params["beta"]) +jax.tree.map(positive,params["epsilon"]))/(jax.tree.map(positive,params["alpha"]) + jax.tree.map(positive,params["gamma"]) + jax.tree.map(positive,params["delta"]) + 1.0))
        )
    status = "converged" if best_metadata["converged"] else "reached max steps"
    print(f"Optimization status: {status} after {best_metadata['n_steps']} step(s)")
    print('Objective function: ', loss_fn(params))
    if fit_replicates > 1:
        plot_replicates(all_params, best_idx)
        plt.show()
    if omega_mode == "per-site":
        fig, _ = plot_per_site_omega(params, mask=mask, domain_labels=domain_labels)
        if output is not None:
            fig.savefig(mode_output_stem(output, omega_mode) + "_omega_plot.pdf",
                        format="pdf", bbox_inches="tight")
        plt.show()
    if estimate_uncertainty:
        logging.info("Computing Laplace uncertainty...")
        _, se_nat = compute_laplace_se(fn, params)
        _print_laplace_summary(params, se_nat, omega_mode=omega_mode)

    if output is not None:
        save_params(output, params, mask=mask, omega_mode=omega_mode)

    return {
        "params": params,
        "objective": float(fn(params)),
        "metadata": best_metadata,
    }
