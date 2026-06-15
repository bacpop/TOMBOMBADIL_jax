#!/usr/bin/env python

import csv as _csv
import logging
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.special as special
from jax.scipy.special import gammaln
import optax
from jax import jit
from jax.flatten_util import ravel_pytree
jax.config.update('jax_enable_x64', True)
import matplotlib.pyplot as plt

from .gtr import build_GTR
from .likelihood import gen_alpha
from .likelihood import gen_alpha_no_jitter


SCALAR_PARAM_KEYS_WITH_ETA = ["alpha", "beta", "gamma", "delta", "epsilon", "eta", "theta", "omega"]

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


def prior_log_likelihood(raw_x, n_sites, prior_mode="current", estimate_eta=False):
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

    if prior_mode in ("stan_constrained", "stan_unconstrained"):
        return omega_prior + gtr_prior + theta_prior

    return (omega_prior + gtr_prior) / n_sites + theta_prior


def make_fn(pi_eq, log_pi, pimat, pimatinv, pimult, X, mask,
            include_invariant=True, aggregate="mean", prior_mode="current",
            estimate_eta=False, eigen_jitter=True, omega_floor=True): # closure for defining fn (this change is mainly for making the unit testing easier, before it was a closure in run_sampler())
    model_fn = model if eigen_jitter else model_no_jitter
    batched_loss = jax.vmap(
        model_fn,
        in_axes=(None, None, None, None, None, None, None, None, None, None, None, None, None, 1)  # map over data only; omega is alignment-wide
    )
    def f(raw_x):

        #x = jnp.exp(x)
        #print('x: ',x)
        #return model(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7:], pi_eq, log_pi, N[col], pimat, pimatinv, pimult, X[:, col])
        #return model(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7:], pi_eq, log_pi, N[col], pimat, pimatinv, pimult, X)
        x = jax.tree.map(positive, raw_x)

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
        total = total + prior_log_likelihood(raw_x, X.shape[1], prior_mode=prior_mode, estimate_eta=estimate_eta)
        return total
    return f


def natural_to_raw_params(params):
    """Convert positive natural-scale parameters to this code's raw log scale."""
    return {k: jnp.array(softplus_inverse(v), dtype=jnp.float64) for k, v in params.items()}


def make_mask(X):
    col_max = np.max(X, axis=0)
    col_sum = np.sum(X, axis=0)
    return np.where(col_max == col_sum, 0, 1)


def evaluate_fixed_params(X, pi_eq, natural_params, include_invariant=True,
                          aggregate="sum", prior_mode="none",
                          estimate_eta=True, eigen_jitter=False,
                          omega_floor=False):
    """Evaluate the scalar-GTR objective at fixed natural-scale parameters."""
    log_pi, pimat, pimatinv, pimult = transforms(X, pi_eq)
    mask = make_mask(X)
    raw_params = natural_to_raw_params(natural_params)
    fn = make_fn(
        pi_eq, log_pi, pimat, pimatinv, pimult, X, mask,
        include_invariant=include_invariant,
        aggregate=aggregate,
        prior_mode=prior_mode,
        estimate_eta=estimate_eta,
        eigen_jitter=eigen_jitter,
        omega_floor=omega_floor,
    )
    return float(fn(raw_params))


def _optimize_params(fn, params, solver, n_iter, verbose=True):
    """Run the optimization loop and return final params."""
    loss_fn = lambda p: -fn(p)
    opt_state = solver.init(params)
    for _ in range(n_iter):
        grad = jax.grad(loss_fn)(params)
        updates, opt_state = solver.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        if verbose:
            print('parameters: ', jax.tree.map(positive, jnp.array([
                params["alpha"], params["beta"], params["gamma"],
                params["delta"], params["epsilon"], params["theta"]
            ])))
            print('omegas: ', jax.tree.map(positive, params["omega"]))
    return params


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


def _print_laplace_summary(params, se_natural):
    """Print a human-readable summary of MAP estimates ± 1 SE (natural scale)."""
    gtr_keys = SCALAR_PARAM_KEYS_WITH_ETA
    print("\n--- Laplace approximation (diagonal) ---")
    print("Scalar parameters (natural scale):")
    for k in gtr_keys:
        if k in params:
            est = float(positive(params[k]))
            se  = float(se_natural[k])
            print(f"  {k:8s}: {est:.4f} ± {se:.4f}")
    print("----------------------------------------\n")


def save_params(output_stem: str, params: dict, mask: np.ndarray = None) -> None:
    """Save MAP scalar parameter estimates to CSV.

    {output_stem}_scalar.csv  — scalar parameters (variable, value)
    """
    scalar_path = output_stem + "_scalar.csv"

    scalar_keys = SCALAR_PARAM_KEYS_WITH_ETA
    rows = [(k, float(positive(params[k]))) for k in scalar_keys if k in params]
    with open(scalar_path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["variable", "value"])
        w.writerows(rows)
    logging.info("Saved scalar parameters to: %s", scalar_path)


def _perturb_params(params, scale=0.5):
    """Add Normal(0, scale) noise to all parameters in raw (unconstrained) space."""
    key = jax.random.PRNGKey(int(np.random.randint(0, 2**31)))
    flat, unflatten = ravel_pytree(params)
    noise = jax.random.normal(key, flat.shape) * scale
    return unflatten(flat + noise)


def _run_replicates(fn, start_params, param_labels, n_reps, n_iter=100):
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
        params_rep = _optimize_params(fn, start, solver, n_iter, verbose=False)
        ll = float(fn(params_rep))
        all_params.append(params_rep)
        all_lls.append(ll)
        logging.info(f"  Replicate {rep + 1}/{n_reps}: log-likelihood = {ll:.4f}")
    best_idx = int(np.argmax(all_lls))
    logging.info(f"Best replicate: {best_idx + 1} (log-likelihood = {all_lls[best_idx]:.4f})")
    return all_params, best_idx


def plot_replicates(all_params_list, best_idx):
    """Plot scalar parameter estimates across replicate runs."""
    n_reps = len(all_params_list)
    scalar_keys = [k for k in SCALAR_PARAM_KEYS_WITH_ETA if k in all_params_list[0]]
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


def run_sampler(X, pi_eq, samples=500, platform='cpu', threads=8,
                estimate_uncertainty=False, fit_replicates=1,
                include_invariant=True, output=None, aggregate="mean",
                prior_mode="current", estimate_eta=False,
                eigen_jitter=True, omega_floor=True):
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

    # Base params and labels shared by both runs
    # eta is fixed to 1.0 inside make_fn(); it is intentionally not part of the
    # sampled parameter set so the global GTR rate scale is identified.
    base_params = {
        "alpha":   jnp.array(softplus_inverse(1),   dtype=jnp.float64),
        "beta":    jnp.array(softplus_inverse(1),   dtype=jnp.float64),
        "gamma":   jnp.array(softplus_inverse(1),   dtype=jnp.float64),
        "delta":   jnp.array(softplus_inverse(1),   dtype=jnp.float64),
        "epsilon": jnp.array(softplus_inverse(1),   dtype=jnp.float64),
        "theta":   jnp.array(softplus_inverse(0.5), dtype=jnp.float64),
        "omega":   jnp.array(softplus_inverse(0.5), dtype=jnp.float64),
    }
    if estimate_eta:
        base_params["eta"] = jnp.array(softplus_inverse(1), dtype=jnp.float64)
    base_labels = {
        "omega": "scalar", "alpha": "scalar", "beta": "scalar", "gamma": "scalar",
        "delta": "scalar", "epsilon": "scalar", "theta": "scalar",
    }
    if estimate_eta:
        base_labels["eta"] = "scalar"

    logging.info(f"Running optimization — {fit_replicates} replicate(s)...")
    fn = make_fn(
        pi_eq, log_pi, pimat, pimatinv, pimult, X, mask,
        include_invariant=include_invariant,
        aggregate=aggregate,
        prior_mode=prior_mode,
        estimate_eta=estimate_eta,
        eigen_jitter=eigen_jitter,
        omega_floor=omega_floor,
    )
    all_params, best_idx = _run_replicates(fn, base_params, base_labels, fit_replicates, n_iter=samples)
    params = all_params[best_idx]

    loss_fn = lambda p: -fn(p)
    print('Final likelihood: ', fn(params))
    print('final parameters: ', jax.tree.map(positive, jnp.array([
        params["alpha"], params["beta"], params["gamma"],
        params["delta"], params["epsilon"], params["theta"], params["omega"]
    ])))
    if estimate_eta:
        print('final eta: ', positive(params["eta"]))
    print('Objective function: ', loss_fn(params))
    if fit_replicates > 1:
        plot_replicates(all_params, best_idx)
        plt.show()
    if estimate_uncertainty:
        logging.info("Computing Laplace uncertainty...")
        _, se_nat = compute_laplace_se(fn, params)
        _print_laplace_summary(params, se_nat)

    if output is not None:
        save_params(output, params)
