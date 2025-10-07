#!/usr/bin/env python

import logging
import numpy as np
import scipy

from .gtr import build_GTR
from .likelihood import gen_alpha

def model(alpha, beta, gamma, delta, epsilon, eta, mu, omega, pi_eq, log_pi, N, pimat, pimatinv, pimult, obs_vec):
    # Calculate substitution rate matrix under neutrality
    #print(pimat)
    #print(pimult)
    A = build_GTR(alpha, beta, gamma, delta, epsilon, eta, 1, pimat, pimult) # 61x61 subst rate matrix
    #A = build_GTR(alpha, beta, gamma, delta, epsilon, eta, omega, pimat, pimult) # 61x61 subst rate matrix # better than fixing omega to 1?
    #print(A) # is all zeros at the moment
    #print(pi_eq)
    meanrate = -np.dot(np.diagonal(A), pi_eq)
    # Calculate substitution rate matrix
    scale = (mu / 2.0) / meanrate

    alpha = gen_alpha(omega, A, pimat, pimult, pimatinv, scale) # does alpha have the right dimensions? I thought it would need to be a vector? or maybe it is just missing values? everything seems to be zero except diagonal
    print('alpha: ',alpha)
    print("obs_vec: ", obs_vec)
    print("N: ", N)
    log_prob = scipy.stats.multinomial.pmf(obs_vec, N, alpha) # this is where it breaks but is it because the code is broken or because of lack of diversity? It is not because of the lack of diversity
    print('log_prob: ',log_prob)
    return scipy.special.logsumexp(log_prob + log_pi, axis=0) # check that these go in as different arguments

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

    return N, n_loci, log_pi, pimat, pimatinv, pimult

def run_sampler(X, pi_eq, warmup=500, samples=500, platform='cpu', threads=8):
    logging.info("Precomputing transforms...")
    #col = 31 # site in the alignment
    col = 7 # site in the alignment # this is a column with a bit of diversity (unlike 31)
    # add for loop later
    N, l, log_pi, pimat, pimatinv, pimult = transforms(X, pi_eq)
    # l is length of alignment

    logging.info("Compiling model...") # jax first compiles code
    def fn(x): return model(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], pi_eq, log_pi, N[col], pimat, pimatinv, pimult, X[:, col])

    # TODO: set threads/device/optim options
    # TODO: work for multiple codons

    # For now: [alpha, beta, gamma, delta, epsilon, eta, theta, omega]
    # 6 parameters of GTR matrix, theta, omega
    params = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    print('Likelihood: ', fn(params))


