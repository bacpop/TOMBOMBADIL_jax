#!/usr/bin/env python

import logging
import gzip
import os
import numpy as np

from .__init__ import __version__

# expected order
# "TTT","TTC","TTA","TTG","TCT","TCC","TCA","TCG","TAT","TAC","TGT","TGC"
# "TGG","CTT","CTC","CTA","CTG","CCT","CCC","CCA","CCG","CAT","CAC","CAA"
# "CAG","CGT","CGC","CGA","CGG","ATT","ATC","ATA","ATG","ACT","ACC","ACA"
# "ACG","AAT","AAC","AAA","AAG","AGT","AGC","AGA","AGG","GTT","GTC","GTA"
# "GTG","GCT","GCC","GCA","GCG","GAT","GAC","GAA","GAG","GGT","GGC","GGA"
# "GGG"
# mapped to
col_order = np.array([63, 61, 60, 62, 55, 53, 52, 54, 51, 49, 59, 57, 58, 31, 29, 28, 30,
                      23, 21, 20, 22, 19, 17, 16, 18, 27, 25, 24, 26, 15, 13, 12, 14,  7,
                       5,  4,  6,  3,  1,  0,  2, 11,  9,  8, 10, 47, 45, 44, 46, 39, 37,
                      36, 38, 35, 33, 32, 34, 43, 41, 40, 42, 48, 50, 56, 64])
#stop codons 48 50 56
#Ns 64

# I think what this means: reading in with the 0,1,2,3 encoding results in order A,C,G,T
# but we want order T,C,A,G
# AAA = 0, AAC = 1, AAG = 2, AAT, 3, 
# ACA = 4, ACC = 5, ACG = 6, ACT = 7, 
# AGA = 8, AGC = 9, AGG = 10, AGT = 11
# ATA = 12, ATC = 13, ATG = 14, ATT = 15
# C** = 16-31
# G** = 32 - 47
# TAA = STOP = 48, TAC = 49, TAG = STOP = 50, TAT = 51
# TCA = 52, TCC = 53, TCG = 54, TCT = 55,
# TGA = STOP = 56, TGC = 57, TGG = 58, TGT = 59
# TTA = 60, TTC = 61, TTG = 62, TTT = 63

BASE_ORDER = ("T", "C", "A", "G")
STOP_CODONS = {"TAA", "TAG", "TGA"}
CODON_LIST = tuple(
    codon
    for codon in (
        first + second + third
        for first in BASE_ORDER
        for second in BASE_ORDER
        for third in BASE_ORDER
    )
    if codon not in STOP_CODONS
)
BASE_TO_INDEX = {base: idx for idx, base in enumerate(BASE_ORDER)}

def get_options():
    import argparse
    parser = argparse.ArgumentParser(description='TOMBOMBADIL (Tree-free Omega Mapping By Observing Mutations of Bases and Amino acids Distributed Inside Loci)',
                                     prog='tombombadil')

    # input options
    iGroup = parser.add_argument_group('Input files')
    iGroup.add_argument('--alignment', type=str, required=True,
                        help='Alignment file to fit model to')

    mGroup = parser.add_argument_group('Model options')
    mGroup.add_argument('--pi', choices=['uniform', 'empirical', 'F3x4'], default='uniform',
                        help='Codon equilibrium frequencies: uniform or estimated from the alignment '
                             '(default: uniform)')
    mGroup.add_argument('--pi-pseudocount', type=float, default=0.5,
                        help='Pseudocount used when --pi empirical or --pi F3x4 is selected '
                             '(default: 0.5)')
    mGroup.add_argument('--omega-mode', choices=['scalar', 'per-site'], default='scalar',
                        help='Estimate one omega for the alignment or one omega per codon site '
                             '(default: scalar).')
    mGroup.add_argument('--domains', type=str, default=None,
                        help='Optional domain JSON for colouring per-site omega plots.')
    mGroup.add_argument('--reference', type=str, default=None,
                        help='Reference protein FASTA required with --domains.')
    mGroup.add_argument('--estimate-uncertainty', action='store_true', default=False,
                        help='Compute per-parameter standard errors via diagonal Laplace approximation '
                             '(Hessian-based). Can be memory-intensive for large alignments.')
    mGroup.add_argument('--fit-replicates', type=int, default=1, metavar='N',
                        help='Run the optimiser N times with random perturbations of the starting values '
                             'and produce a convergence plot. Best replicate (highest log-likelihood) is used '
                             'for all downstream outputs (default: 1).')
    mGroup.add_argument('--exclude-invariant', action='store_true', default=False,
                        help='Exclude invariant sites from the data likelihood. By default invariant '
                             'sites are included.')
    mGroup.add_argument('--output-jax', type=str, default=None, metavar='STEM',
                        help='Save MAP or NUTS output files with the given stem. Default: do not save.')
    mGroup.add_argument('--fit-method', choices=['map', 'nuts'], default='map',
                        help='Fit with MAP optimisation or BlackJAX NUTS sampling (default: map).')
    mGroup.add_argument('--objective-aggregate', choices=['mean', 'sum'], default='sum',
                        help='Aggregate site log-likelihoods by mean or sum (default: sum).')
    mGroup.add_argument('--prior-mode',
                        choices=['current', 'none', 'stan_constrained', 'stan_unconstrained'],
                        default='stan_unconstrained',
                        help='Prior/Jacobian convention for optimisation (default: stan_unconstrained).')
    mGroup.add_argument('--fix-eta', action='store_true', default=False,
                        help='Fix eta to 1.0 instead of estimating it.')
    mGroup.add_argument('--fit-until-convergence', action='store_true', default=False,
                        help='Stop optimisation early when the objective stops improving.')
    mGroup.add_argument('--convergence-tol', type=float, default=1e-6,
                        help='Minimum objective improvement counted as progress (default: 1e-6).')
    mGroup.add_argument('--convergence-patience', type=int, default=5,
                        help='Number of convergence checks without progress before stopping (default: 5).')
    mGroup.add_argument('--convergence-check-every', type=int, default=10,
                        help='Check convergence every N optimiser steps (default: 10).')
    mGroup.add_argument('--convergence-min-steps', type=int, default=50,
                        help='Minimum optimiser steps before convergence can stop fitting (default: 50).')
    mGroup.add_argument('--num-warmup', type=int, default=1000,
                        help='Number of BlackJAX NUTS warmup steps per chain (default: 1000).')
    mGroup.add_argument('--num-samples', type=int, default=1000,
                        help='Number of BlackJAX NUTS posterior draws per chain (default: 1000).')
    mGroup.add_argument('--num-chains', type=int, default=4,
                        help='Number of BlackJAX NUTS chains (default: 4).')
    mGroup.add_argument('--rng-seed', type=int, default=0,
                        help='Random seed for BlackJAX NUTS (default: 0).')
    mGroup.add_argument('--target-acceptance-rate', type=float, default=0.8,
                        help='Target acceptance rate for BlackJAX window adaptation (default: 0.8).')

    dGroup = parser.add_argument_group('Diagnostic fixed-parameter scoring')
    dGroup.add_argument('--diagnostic-fixed-params', action='store_true', default=False,
                        help='Evaluate the scalar-GTR objective at fixed natural-scale parameters and exit.')
    dGroup.add_argument('--diagnostic-alpha', type=float, default=1.0)
    dGroup.add_argument('--diagnostic-beta', type=float, default=1.0)
    dGroup.add_argument('--diagnostic-gamma', type=float, default=1.0)
    dGroup.add_argument('--diagnostic-delta', type=float, default=1.0)
    dGroup.add_argument('--diagnostic-epsilon', type=float, default=1.0)
    dGroup.add_argument('--diagnostic-eta', type=float, default=1.0)
    dGroup.add_argument('--diagnostic-theta', type=float, default=0.5)
    dGroup.add_argument('--diagnostic-omega', type=float, default=0.5)
    dGroup.add_argument('--diagnostic-prior-mode',
                        choices=['none', 'current', 'stan_constrained', 'stan_unconstrained'],
                        default='none',
                        help='Prior/Jacobian convention for fixed scoring (default: none).')
    dGroup.add_argument('--diagnostic-aggregate', choices=['sum', 'mean'], default='sum',
                        help='Aggregate site log-likelihoods for fixed scoring (default: sum).')
    dGroup.add_argument('--diagnostic-fix-eta', action='store_true', default=False,
                        help='Score with eta fixed to 1.0 instead of using --diagnostic-eta.')
    dGroup.add_argument('--diagnostic-enable-jitter', action='store_true', default=False,
                        help='Enable the JAX eigen jitter while fixed scoring (default: disabled).')
    dGroup.add_argument('--diagnostic-enable-omega-floor', action='store_true', default=False,
                        help='Enable the JAX omega gradient floor while fixed scoring (default: disabled).')

    sGroup = parser.add_argument_group('Sampling options')
    sGroup.add_argument('--sample-it', type=int, default=500,
                        help='Sampling iterations')

    hGroup = parser.add_argument_group('Hardware options')
    hGroup.add_argument('--platform', choices=['cpu', 'gpu', 'tpu'], default='cpu',
                        help='Which hardware/device to run on')
    hGroup.add_argument('--cpus', type=int, default=1,
                        help='Number of CPU cores to use')
    hGroup.add_argument('--nuts-chain-mode', choices=['sequential', 'pmap'], default='sequential',
                        help='Run NUTS chains sequentially or in parallel across JAX devices '
                             '(default: sequential).')

    other = parser.add_argument_group('Other options')
    other.add_argument('--version', action='version',
                       version='%(prog)s '+__version__)

    args = parser.parse_args()
    return args

def read_fasta(fp):
    name, seq = None, []
    for line in fp:
        line = line.rstrip()
        if line.startswith(">"):
            if name: yield (name, ''.join(seq))
            name, seq = line[1:], []
        else:
            seq.append(line)
    if name: yield (name, ''.join(seq))

def count_codons(file_name):
    n_samples = 0
    with open(file_name, 'rb') as test_f:
        zipped = test_f.read(2) == b'\x1f\x8b'
    if zipped:
        fh = gzip.open(file_name, 'rt')
    else:
        fh = open(file_name, 'rt')
    with fh as fasta:
        X = None
        for h, s in read_fasta(fasta):
            n_samples += 1
            s = np.frombuffer(s.lower().encode(), dtype=np.int8)
            if X is None:
                X = np.zeros((65, s.shape[0] // 3), dtype=np.int32)
            # Set ambiguous bases
            ambig = np.argwhere((s!=97) & (s!=99) & (s!=103) & (s!=116))
            #print("ambig",ambig)
            s = np.copy(s) # without copying I got ValueError: assignment destination is read-only
            if ambig.any():
                s[ambig] = 64
            codon_s = s.reshape(-1, 3).copy()
            #print('codon_s',codon_s)
            # Convert to usual binary encoding
            codon_s[codon_s==97] = 0 # A
            codon_s[codon_s==99] = 1 # C
            codon_s[codon_s==103] = 2 # G
            codon_s[codon_s==116] = 3 # T
            # Bit shift
            #print('codon_s',codon_s)
            codon_s[:,1] = np.left_shift(codon_s[:, 1], 2)
            codon_s[:,0] = np.left_shift(codon_s[:, 0], 4) # changed bit shift to first position (because we're ordering AAA, AAC, AAG, AAT, ACA, ... (= first position has longest "duration"))
            codon_map = np.fmin(np.sum(codon_s, 1), 64)
            #print('codon_s',codon_s)
            #print('codon_map',codon_map)
            # slow? Alternative would be to make X have shape (samples, n_codons)
            # and copy codon map into each row, then run np.bincount along columns
            for idx, count in enumerate(codon_map):
                X[count,idx] += 1

    # reorder and cut off stops, ambiguous
    #print("X", X[:,10])
    X = X[col_order,:]
    X = X[0:61, :]

    return X, n_samples

def estimate_pi_from_counts(X, pseudocount=0.5):
    X = np.asarray(X)
    if X.shape[0] != 61:
        raise ValueError(f"Expected codon count matrix with 61 rows, got {X.shape[0]}")
    if pseudocount < 0:
        raise ValueError("--pi-pseudocount must be non-negative")

    observed_counts = X.sum(axis=1, dtype=np.float64)
    if observed_counts.sum() <= 0:
        raise ValueError("Cannot estimate pi: no non-stop codons were observed in the alignment")

    smoothed_counts = observed_counts + pseudocount
    total = smoothed_counts.sum()
    if total <= 0:
        raise ValueError("Cannot estimate pi: smoothed codon counts sum to zero")

    pi = smoothed_counts / total
    if pi.shape != (61,) or not np.all(np.isfinite(pi)) or np.any(pi <= 0):
        raise ValueError(
            "Estimated pi must contain 61 finite, strictly positive non-stop codon frequencies"
        )
    print("pi",pi)
    return pi

def estimate_f3x4_frequencies_from_counts(X, pseudocount=0.5):
    X = np.asarray(X)
    if X.shape[0] != 61:
        raise ValueError(f"Expected codon count matrix with 61 rows, got {X.shape[0]}")
    if pseudocount < 0:
        raise ValueError("--pi-pseudocount must be non-negative")

    observed_counts = X.sum(axis=1, dtype=np.float64)
    if observed_counts.sum() <= 0:
        raise ValueError("Cannot estimate F3x4 pi: no non-stop codons were observed in the alignment")

    nucleotide_counts = np.full((3, 4), pseudocount, dtype=np.float64)
    for codon, count in zip(CODON_LIST, observed_counts):
        for position, base in enumerate(codon):
            nucleotide_counts[position, BASE_TO_INDEX[base]] += count

    row_totals = nucleotide_counts.sum(axis=1, keepdims=True)
    if np.any(row_totals <= 0):
        raise ValueError("Cannot estimate F3x4 pi: smoothed nucleotide counts sum to zero")

    frequencies = nucleotide_counts / row_totals
    if frequencies.shape != (3, 4) or not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0):
        raise ValueError(
            "Estimated F3x4 nucleotide frequencies must be a finite, strictly positive 3x4 matrix"
        )

    return frequencies

def estimate_f3x4_pi_from_counts(X, pseudocount=0.5):
    frequencies = estimate_f3x4_frequencies_from_counts(X, pseudocount)
    pi = np.array(
        [
            frequencies[0, BASE_TO_INDEX[codon[0]]]
            * frequencies[1, BASE_TO_INDEX[codon[1]]]
            * frequencies[2, BASE_TO_INDEX[codon[2]]]
            for codon in CODON_LIST
        ],
        dtype=np.float64,
    )
    total = pi.sum()
    if total <= 0:
        raise ValueError("Cannot estimate F3x4 pi: codon frequencies sum to zero")

    pi = pi / total
    if pi.shape != (61,) or not np.all(np.isfinite(pi)) or np.any(pi <= 0):
        raise ValueError("Estimated F3x4 pi must contain 61 finite, strictly positive codon frequencies")

    return pi

def configure_jax_for_options(options):
    """Set JAX process flags that must exist before JAX is imported."""
    from .device import configure_platform
    configure_platform(
        options.platform,
        cpus=options.cpus,
        force_cpu_devices=(
            options.fit_method == "nuts" and options.nuts_chain_mode == "pmap"
        ),
    )

def main():
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True)

    options = get_options()
    if options.cpus != 1 and options.nuts_chain_mode != "pmap":
        logging.warning(
            "--cpus=%s has no effect unless --nuts-chain-mode pmap is used; "
            "MAP optimisation and sequential NUTS chains run serially.",
            options.cpus,
        )
    configure_jax_for_options(options)
    logging.info("Reading alignment...")
    X, n_samples = count_codons(options.alignment)
    logging.info(f"Read {n_samples} samples and {X.shape[1]} codons")

    #print("X",X.max())
    if options.pi == 'uniform':
        pi = np.full(61, 1 / 61)
        logging.info("Using uniform codon equilibrium frequencies")
    elif options.pi == 'empirical':
        pi = estimate_pi_from_counts(X, options.pi_pseudocount)
        logging.info(
            "Using empirical codon equilibrium frequencies estimated from alignment "
            f"with pseudocount {options.pi_pseudocount}"
        )
    elif options.pi == 'F3x4':
        pi = estimate_f3x4_pi_from_counts(X, options.pi_pseudocount)
        logging.info(
            "Using F3x4 codon equilibrium frequencies estimated from alignment "
            f"with pseudocount {options.pi_pseudocount}"
        )
    else:
        raise ValueError(f"Unsupported --pi mode: {options.pi}")

    if options.diagnostic_fixed_params:
        if options.omega_mode != 'scalar':
            raise ValueError('--diagnostic-fixed-params currently supports only --omega-mode scalar')
        from .sample import evaluate_fixed_params

        diagnostic_params = {
            "alpha": options.diagnostic_alpha,
            "beta": options.diagnostic_beta,
            "gamma": options.diagnostic_gamma,
            "delta": options.diagnostic_delta,
            "epsilon": options.diagnostic_epsilon,
            "eta": options.diagnostic_eta,
            "theta": options.diagnostic_theta,
            "omega": options.diagnostic_omega,
        }
        value = evaluate_fixed_params(
            X, pi, diagnostic_params,
            include_invariant=not options.exclude_invariant,
            aggregate=options.diagnostic_aggregate,
            prior_mode=options.diagnostic_prior_mode,
            estimate_eta=not options.diagnostic_fix_eta,
            eigen_jitter=options.diagnostic_enable_jitter,
            omega_floor=options.diagnostic_enable_omega_floor,
            omega_mode=options.omega_mode,
        )
        print(f"Diagnostic scalar-GTR objective: {value:.10f}")
        return

    from .sample import run_sampler

    domain_labels = None
    if options.domains is not None:
        if options.omega_mode != 'per-site':
            raise ValueError('--domains requires --omega-mode per-site')
        if options.reference is None:
            raise ValueError('--reference is required when --domains is specified')
        from .domains import parse_domain_labels
        domain_labels = parse_domain_labels(options.domains, options.alignment,
                                            options.reference, X.shape[1])

    run_sampler(X, pi, options.sample_it, options.platform, options.cpus,
                estimate_uncertainty=options.estimate_uncertainty,
                fit_replicates=options.fit_replicates,
                include_invariant=not options.exclude_invariant,
                output=options.output_jax,
                aggregate=options.objective_aggregate,
                prior_mode=options.prior_mode,
                estimate_eta=not options.fix_eta,
                fit_until_convergence=options.fit_until_convergence,
                convergence_tol=options.convergence_tol,
                convergence_patience=options.convergence_patience,
                convergence_check_every=options.convergence_check_every,
                convergence_min_steps=options.convergence_min_steps,
                fit_method=options.fit_method,
                num_warmup=options.num_warmup,
                num_samples=options.num_samples,
                num_chains=options.num_chains,
                rng_seed=options.rng_seed,
                target_acceptance_rate=options.target_acceptance_rate,
                nuts_chain_mode=options.nuts_chain_mode,
                omega_mode=options.omega_mode,
                domain_labels=domain_labels)

if __name__ == "__main__":
    main()
