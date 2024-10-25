#!/usr/bin/env python

import logging
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.lax
import jax.numpy as jnp
import jax.random as random
from jax import vmap

from .gtr import build_GTR
from .likelihood import gen_alpha
vmap_gen_alpha = vmap(gen_alpha, (0, None, None, None, None, None), 0)

def model(pi_eq, N, l, pimat, pimatinv, pimult, obs_mat):
    # GTR params (shared)
    alpha = numpyro.sample("alpha", dist.LogNormal(1, 1))
    beta = numpyro.sample("beta", dist.LogNormal(1, 1))
    gamma = numpyro.sample("gamma", dist.LogNormal(1, 1))
    delta = numpyro.sample("delta", dist.LogNormal(1, 1))
    epsilon = numpyro.sample("epsilon", dist.LogNormal(1, 1))
    eta = numpyro.sample("eta", dist.LogNormal(1, 1))

    # Rate params
    theta = numpyro.sample("theta", dist.Gamma(1, 2))

    # Calculate substitution rate matrix under neutrality
    A = build_GTR(alpha, beta, gamma, delta, epsilon, eta, 1, pimat, pimult)
    meanrate = -jnp.dot(jnp.diagonal(A), pi_eq)
    # Calculate substitution rate matrix
    scale = (theta / 2.0) / meanrate

    # Over loci
    loci = numpyro.plate('locus', l, dim=-2)
    with loci:
        omega = numpyro.sample("omega", dist.Exponential(0.5))
        alpha = vmap_gen_alpha(omega, A, pimat, pimult, pimatinv, scale)

    log_prob = numpyro.contrib.control_flow.scan(count_prob, 0, alpha, length=61, reverse=False, history=0)
    log_prob = logsumexp(log_prob + log_pi, axis=0, keepdims=True) # log pi probably needs broadcasting
    numpyro.factor("forward_log_prob", log_prob)

# TODO need a closure so signature is f(carry, alpha)
def count_prob(loci, alpha, N, obs_mat):
    with loci:
        obs = jnp.log(numpyro.sample('obs', dist.DirichletMultinomial(concentration=alpha, total_count=N), obs=obs_mat))
    return obs

# TODO the sparse multinomial

def transforms(X, pi_eq):
    import numpy as np
    N = np.sum(X, 0)
    n_loci = len(N)

    # pi transforms
    pimat = np.diag(np.sqrt(pi_eq))
    pimatinv = np.diag(np.divide(1, np.sqrt(pi_eq)))

    pimult = np.zeros((61, 61))
    for j in range(61)  :
        for i in range(61):
            pimult[i, j] = np.sqrt(pi_eq[j] / pi_eq[i])

    obs_mat = np.empty((n_loci, 61, 61))
    N_tile = np.empty((n_loci, 61))
    for l in range(n_loci):
        obs_mat[l, :, :] = np.broadcast_to(X[:, l], (61, 61))
        N_tile[l, :] = N[l]

    return N_tile, n_loci, pimat, pimatinv, pimult, obs_mat

def run_sampler(X, pi_eq, warmup=500, samples=500, platform='cpu', threads=8):
    logging.info("Precomputing transforms...")
    N, l, pimat, pimatinv, pimult, obs_mat = transforms(X, pi_eq)

    logging.info("Compiling model...")
    numpyro.set_platform(platform)
    # this doesn't do what you might think, it's more like having multiple GPUs
    #numpyro.set_host_device_count(threads)
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=warmup, num_samples=samples)
    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key, pi_eq, N, l, pimat, pimatinv, pimult, obs_mat,
             extra_fields=('potential_energy',), chain_method="vectorized")
    mcmc.print_summary()
    pe = np.mean(-mcmc.get_extra_fields()['potential_energy'])
    print(f'Expected log joint density: {pe}')


