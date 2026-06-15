#!/usr/bin/env python

import logging
import gzip
import numpy as np

from .__init__ import __version__
from .sample import evaluate_fixed_params
from .sample import run_sampler

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

def get_options():
    import argparse
    parser = argparse.ArgumentParser(description='TOMBOMBADIL (Tree-free Omega Mapping By Observing Mutations of Bases and Amino acids Distributed Inside Loci)',
                                     prog='tombombadil')

    # input options
    iGroup = parser.add_argument_group('Input files')
    iGroup.add_argument('--alignment', type=str, required=True,
                        help='Alignment file to fit model to')

    mGroup = parser.add_argument_group('Model options')
    mGroup.add_argument('--pi', type=str, default=None,
                        help='Pi equilibrium vector (default all equal)')
    mGroup.add_argument('--estimate-uncertainty', action='store_true', default=False,
                        help='Compute per-parameter standard errors via diagonal Laplace approximation '
                             '(Hessian-based). Can be memory-intensive for large alignments.')
    mGroup.add_argument('--fit-replicates', type=int, default=1, metavar='N',
                        help='Run the optimiser N times with random perturbations of the starting values '
                             'and produce a convergence plot. Best replicate (highest log-likelihood) is used '
                             'for all downstream outputs (default: 1).')
    mGroup.add_argument('--exclude-invariant', action='store_true', default=False,
                        help='Exclude invariant sites from the mean data likelihood. By default invariant '
                             'sites are included.')
    mGroup.add_argument('--output-jax', type=str, default=None, metavar='STEM',
                        help='Save MAP estimates to STEM_scalar.csv. Default: do not save.')
    mGroup.add_argument('--objective-aggregate', choices=['mean', 'sum'], default='mean',
                        help='Aggregate site log-likelihoods by mean or sum (default: mean).')
    mGroup.add_argument('--prior-mode',
                        choices=['current', 'none', 'stan_constrained', 'stan_unconstrained'],
                        default='current',
                        help='Prior/Jacobian convention for optimisation (default: current).')
    mGroup.add_argument('--estimate-eta', action='store_true', default=False,
                        help='Estimate eta instead of fixing eta to 1.0.')
    mGroup.add_argument('--disable-eigen-jitter', action='store_true', default=False,
                        help='Disable the 1e-6 diagonal jitter before eigendecomposition.')
    mGroup.add_argument('--disable-omega-floor', action='store_true', default=False,
                        help='Disable the omega <= 0.01 gradient stop used by the default optimiser.')

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
    sGroup.add_argument('--platform', choices=['cpu', 'gpu', 'tpu'], default='cpu',
                        help='Which hardware/device to run on')
    sGroup.add_argument('--cpus', type=int, default=8,
                        help='Number of CPU cores to use')

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

def main():
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True)

    options = get_options()
    logging.info("Reading alignment...")
    X, n_samples = count_codons(options.alignment)
    logging.info(f"Read {n_samples} samples and {X.shape[1]} codons")

    #print("X",X.max())
    if options.pi is None:
        pi = np.array([1/61 for i in range(61)])

    if options.diagnostic_fixed_params:
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
        )
        print(f"Diagnostic scalar-GTR objective: {value:.10f}")
        return

    run_sampler(X, pi, options.sample_it, options.platform, options.cpus,
                estimate_uncertainty=options.estimate_uncertainty,
                fit_replicates=options.fit_replicates,
                include_invariant=not options.exclude_invariant,
                output=options.output_jax,
                aggregate=options.objective_aggregate,
                prior_mode=options.prior_mode,
                estimate_eta=options.estimate_eta,
                eigen_jitter=not options.disable_eigen_jitter,
                omega_floor=not options.disable_omega_floor)

if __name__ == "__main__":
    main()
