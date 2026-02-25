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
from jax import jit

from .gtr import build_GTR
from .likelihood import gen_alpha

@jit
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
        return jnp.exp(a)

def softplus_inverse(y, eps=1e-6): # inverse transformation for calculating raw parameter values (e.g. for start values of parameters)
    z = y
    return jnp.log((z))
    
def make_fn(pi_eq, log_pi, pimat, pimatinv, pimult, X, mask): # closure for defining fn (this change is mainly for making the unit testing easier, before it was a closure in run_sampler())
    batched_loss = jax.vmap(
        model,
        in_axes=(None, None, None, None, None, None, None, 0, None, None, None, None, None, 1)  # map over matrices + data
    )
    def f(raw_x): 
        
        #x = jnp.exp(x)
        #print('x: ',x)
        #return model(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7:], pi_eq, log_pi, N[col], pimat, pimatinv, pimult, X[:, col])
        #return model(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7:], pi_eq, log_pi, N[col], pimat, pimatinv, pimult, X)
        x = jax.tree.map(positive, raw_x)

        x["omega"] = jnp.where(
            mask == 1,
            x["omega"],
            jax.lax.stop_gradient(x["omega"])
        )

        losses = batched_loss(x["alpha"], x["beta"], x["gamma"], x["delta"], x["epsilon"], x["eta"], x["theta"], x["omega"], pi_eq, log_pi, pimat, pimatinv, pimult, X)
        #print('losses: ',losses)
        return jnp.mean(losses)
    return f

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
    log_pi, pimat, pimatinv, pimult = transforms(X, pi_eq)
    # l is length of alignment
    print("X",X)
    #print("sum X", np.sum(X))
    #print("larger zero", np.where(X > 0))

    # calculate mask for masking parts of the alignment where there is no diversity
    # this will allow using these position for calculating gradient for constant parameters but excludes omega calculation for these positions
    col_max = np.max(X, axis=0) # finds maximum value per column in X
    col_sum = np.sum(X, axis=0) # calculates column sum
    mask = np.where(col_max == col_sum, 0, 1) # create mask for positions without diversity
    #print("col_max",col_max)
    #print("col_sum",col_sum)
    #print("mask",mask)
    #mask = mask.at[0].set(0.0)

    logging.info("Compiling model...") # jax first compiles code

    fn = make_fn(pi_eq, log_pi, pimat, pimatinv, pimult, X, mask)
    
    # TODO: set threads/device/optim options
    # TODO: work for multiple codons

    # For now: [alpha, beta, gamma, delta, epsilon, eta, theta, omega]
    # 6 parameters of GTR matrix, theta, omega
    #params = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    #params = jnp.array([1, 1, 1, 1, 1, 1, 0.5, 0.5]) # define start parameters for optimization
    params = { # define parameters as dictionary to allow flexible (data-informed)size for omega
        "alpha": jnp.array(softplus_inverse(1), dtype=jnp.float32),
        "beta": jnp.array(softplus_inverse(1), dtype=jnp.float32),
        "gamma": jnp.array(softplus_inverse(1), dtype=jnp.float32),
        "delta": jnp.array(softplus_inverse(1), dtype=jnp.float32),
        "epsilon": jnp.array(softplus_inverse(1), dtype=jnp.float32),
        "eta": jnp.array(softplus_inverse(1), dtype=jnp.float32),
        "theta": jnp.array(softplus_inverse(0.5), dtype=jnp.float32),
        "omega": jnp.repeat(jnp.array(softplus_inverse(0.5), dtype=jnp.float32), jnp.size(X, axis=1)),
    }
    #print('Parameters: ',((params)))

    #params = jnp.array([0, 0, 0, 0, 0, 0, -0.6931472, -0.6931472]) # used this in comibnation of the x = jnp.exp(x) in fn(x) - transformation of parameters but might not be necessary?

    #solver = optax.adabelief(learning_rate=0.003) # adabelief optimizer
    #solver = optax.adam(learning_rate=0.0000001) # adam optimizer
    #solver = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-2)) # define optimizer (adam, with clipping)
    solver = optax.multi_transform(
    {
        "vec": optax.adam(1e-1),
        "scalar": optax.adam(1e-1),
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
    },
)

    def loss(p): return -fn(p) # define loss function (will be minimized that is why it must be -fn(p))
    
    logging.info("Fitting model...")
    opt_state = solver.init(params)

    for _ in range(100): # define number of iterations of optimizer
        grad = jax.grad(loss)(params) # compute gradient
        updates, opt_state = solver.update(grad, opt_state, params) # update states
        params = optax.apply_updates(params, updates) # update parameters
        print('Objective function: ',(loss(params)))
        print('updates: ',(updates))
        print('parameters: ', jax.tree.map(positive, jnp.array([params["alpha"], params["beta"], params["gamma"], params["delta"], params["epsilon"], params["eta"], params["theta"]])))
        #print('raw parameters: ', jnp.array([params["alpha"], params["beta"], params["gamma"], params["delta"], params["epsilon"], params["eta"], params["theta"]]))
        print('omegas: ', jax.tree.map(positive, params["omega"]))
        #print('raw omegas: ', params["omega"])

    print('Final likelihood: ', fn(params)) # print final likelihood
    #print('Final parameters: ',((params)))
    #print('final parameters: ', jnp.array([params["alpha"], params["beta"], params["gamma"], params["delta"], params["epsilon"], params["eta"], params["theta"]]))
    #print('final omega: ', params["omega"])
    #print('Final omega: ',params["omega"][:10]) # only print first ten elements of omega parameters
    print('Objective function: ',(loss(params)))


