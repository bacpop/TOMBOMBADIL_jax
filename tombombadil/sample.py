#!/usr/bin/env python

import logging
import numpy as np
import scipy
import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import jax.scipy.special as special
from jax.scipy.special import gammaln
import optax

from .gtr import build_GTR
from .likelihood import gen_alpha

def my_dirichlet_multinomial_logpmf(x, a):
    x = jnp.asarray(x)
    a = jnp.asarray(a)

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
    #test = gammaln(x)
    #jax.debug.print("test = {test}", test=test)
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

def model(alpha, beta, gamma, delta, epsilon, eta, mu, omega, pi_eq, log_pi, N, pimat, pimatinv, pimult, obs_vec):
    # Calculate substitution rate matrix under neutrality
    #print(pimat)
    #print(pimult)
    #A = build_GTR(alpha, beta, gamma, delta, epsilon, eta, 1, pimat, pimult) # 61x61 subst rate matrix
    #A = build_GTR(1, 1, 1, 1, 1, 1, 1, pimat, pimult) # same as NY98?
    A = build_GTR(alpha, beta, gamma, delta, epsilon, eta, omega, pimat, pimult) # 61x61 subst rate matrix
    #print(A) # is all zeros at the moment
    #print(pi_eq)
    #print(jnp.diagonal(A))
    #print(-jnp.dot(jnp.diagonal(A), pi_eq))
    meanrate = -jnp.dot(jnp.diagonal(A), pi_eq)
    # Calculate substitution rate matrix
    scale = (mu / 2.0) / meanrate

    alpha = gen_alpha(omega, A, pimat, pimult, pimatinv, scale)
    #print('alpha: ',alpha)
    #print("obs_vec: ", obs_vec)
    #print("N: ", N)
    #print(np.sum(alpha,axis=1).tolist()) # alpha rows clearly do not sum to one but this is what the pmf is expecting -- a problem? no, for dirichlet not a problem
    #log_prob = scipy.stats.multinomial.pmf(obs_vec, N, alpha) # this is where it breaks but is it because the code is broken or because of lack of diversity? It is not because of the lack of diversity
    #log_prob = scipy.stats.multinomial.logpmf(obs_vec, N, alpha) # this is pmf in John's code but we think it might need to be pmf?
    # log_prob = scipy.stats.dirichlet_multinomial.logpmf(obs_vec, alpha, N) # gives -10.21301 (correct)
    log_prob = my_dirichlet_multinomial_logpmf(obs_vec, alpha) # our custom, jnp based dirichlet_multinomial.logpmf but something is wrong in the implementation this function gives us an integer, we want a vector of length 61

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
            #pimult = pimult.at[i,j].set(jnp.sqrt(pi_eq[j] / pi_eq[i]))

    return N, n_loci, log_pi, pimat, pimatinv, pimult

def run_sampler(X, pi_eq, warmup=500, samples=500, platform='cpu', threads=8):
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
    N, l, log_pi, pimat, pimatinv, pimult = transforms(X, pi_eq)
    # l is length of alignment
    #print("X",X)

    logging.info("Compiling model...") # jax first compiles code

    batched_loss = jax.vmap(
        model,
        in_axes=(None, None, None, None, None, None, None, 0, None, None, None, None, None, None, 1)  # map over matrices + data
    )

    #def fn(x): return model(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], pi_eq, log_pi, N[col], pimat, pimatinv, pimult, X[:, col])
    def fn(x): 
        #x = jnp.exp(x)
        #print('x: ',x)
        #return model(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7:], pi_eq, log_pi, N[col], pimat, pimatinv, pimult, X[:, col])
        #return model(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7:], pi_eq, log_pi, N[col], pimat, pimatinv, pimult, X)
        losses = batched_loss(x["alpha"], x["beta"], x["gamma"], x["delta"], x["epsilon"], x["eta"], x["theta"], x["omega"], pi_eq, log_pi, N[col], pimat, pimatinv, pimult, X)
        #print('losses: ',losses)
        return jnp.mean(losses)
    
    # TODO: set threads/device/optim options
    # TODO: work for multiple codons

    # For now: [alpha, beta, gamma, delta, epsilon, eta, theta, omega]
    # 6 parameters of GTR matrix, theta, omega
    #params = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    #params = jnp.array([1, 1, 1, 1, 1, 1, 0.5, 0.5]) # define start parameters for optimization
    params = { # define parameters as dictionary to allow flexible (data-informed)size for omega
        "alpha": jnp.array(1.0, dtype=jnp.float32),
        "beta": jnp.array(1.0, dtype=jnp.float32),
        "gamma": jnp.array(1.0, dtype=jnp.float32),
        "delta": jnp.array(1.0, dtype=jnp.float32),
        "epsilon": jnp.array(1.0, dtype=jnp.float32),
        "eta": jnp.array(1.0, dtype=jnp.float32),
        "theta": jnp.array(0.5, dtype=jnp.float32),
        "omega": jnp.repeat(jnp.array(0.5, dtype=jnp.float32), jnp.size(X, axis=1)),
    }
    print('Parameters: ',((params)))

    #params = jnp.array([0, 0, 0, 0, 0, 0, -0.6931472, -0.6931472]) # used this in comibnation of the x = jnp.exp(x) in fn(x) - transformation of parameters but might not be necessary?

    #solver = optax.adabelief(learning_rate=0.003) # adabelief optimizer
    #solver = optax.adam(learning_rate=0.0000001) # adam optimizer
    #solver = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-2)) # define optimizer (adam, with clipping)
    solver = optax.multi_transform(
    {
        "vec": optax.adam(1e-5),
        "scalar": optax.adam(1e-3),
    },
    param_labels={
        "omega": "vec",
        "alpha": "scalar",
        "beta": "scalar",
        "gamma": "scalar",
        "delta": "scalar",
        "epsilon": "scalar",
        "eta": "scalar",
        "theta": "scalar",
        "omega": "scalar",
    },
)

    def loss(p): return -fn(p) # define loss function (will be minimized that is why it must be -fn(p))
    
    logging.info("Fitting model...")
    opt_state = solver.init(params)
    for _ in range(20): # define number of iterations of optimizer
        grad = jax.grad(loss)(params) # compute gradient
        #print('Gradient: ',((grad)))
        updates, opt_state = solver.update(grad, opt_state, params) # update states
        params = optax.apply_updates(params, updates) # update parameters
        print('updates: ',((updates)))
        print('Parameters: ',((params)))
        print('Objective function: ',(loss(params)))

    print('Final likelihood: ', fn(params)) # print final likelihood


