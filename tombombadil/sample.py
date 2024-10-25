#!/usr/bin/env python

import logging
import optax
import jax
import jax.numpy as jnp
import jax.lax
import jax.numpy as jnp

from .gtr import build_GTR
from .likelihood import gen_alpha

@jax.jit
def model(alpha, beta, gamma, delta, epsilon, eta, theta, omega, pi_eq, log_pi, N, pimat, pimatinv, pimult, obs_vec):
    # Calculate substitution rate matrix under neutrality
    A = build_GTR(alpha, beta, gamma, delta, epsilon, eta, 1, pimat, pimult)
    meanrate = -jnp.dot(jnp.diagonal(A), pi_eq)
    # Calculate substitution rate matrix
    scale = (theta / 2.0) / meanrate

    alpha = gen_alpha(omega, A, pimat, pimult, pimatinv, scale)

    log_prob = jax.scipy.stats.multinomial.pmf(obs_vec, N, alpha)
    return jax.scipy.special.logsumexp(log_prob + log_pi, axis=0)

def transforms(X, pi_eq):
    import numpy as np
    N = np.sum(X, 0)
    n_loci = len(N)

    # pi transforms
    log_pi = np.log(pi_eq)
    pimat = np.diag(np.sqrt(pi_eq))
    pimatinv = np.diag(np.divide(1, np.sqrt(pi_eq)))

    pimult = np.zeros((61, 61))
    for j in range(61)  :
        for i in range(61):
            pimult[i, j] = np.sqrt(pi_eq[j] / pi_eq[i])

    #obs_mat = np.empty((n_loci, 61, 61))
    #N_tile = np.empty((n_loci, 61), dtype=np.int32)
    #for l in range(n_loci):
    #    obs_mat[l, :, :] = np.broadcast_to(X[:, l], (61, 61))
    #    N_tile[l, :] = N[l]

    #return N_tile, n_loci, log_pi, pimat, pimatinv, pimult, obs_mat
    return N, n_loci, log_pi, pimat, pimatinv, pimult

def run_sampler(X, pi_eq, warmup=500, samples=500, platform='cpu', threads=8):
    logging.info("Precomputing transforms...")
    col = 31
    N, l, log_pi, pimat, pimatinv, pimult = transforms(X, pi_eq)

    logging.info("Compiling model...")
    def fn(x): return model(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], pi_eq, log_pi, N[col], pimat, pimatinv, pimult, X[:, col])

    # TODO: set threads/device/optim options
    # TODO: work for multiple codons
    solver = optax.lbfgs()

    # For now: [alpha, beta, gamma, delta, epsilon, eta, theta, omega]
    params = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    print('Likelihood: ', fn(params))
    opt_state = solver.init(params)
    value_and_grad = optax.value_and_grad_from_state(fn)

    for _ in range(5):
        value, grad = value_and_grad(params, state=opt_state)
        updates, opt_state = solver.update(
        grad, opt_state, params, value=value, grad=grad, value_fn=fn
        )

        params = optax.apply_updates(params, updates)
        print('Likelihood: ', fn(params))



