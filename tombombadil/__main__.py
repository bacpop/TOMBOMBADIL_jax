#!/usr/bin/env python

import logging
import gzip
import csv
import numpy as np
import matplotlib.pyplot as plt

from .__init__ import __version__
from .domains import parse_domain_json
from .device import configure_platform
from .genetic_code import AA_LIST, CODON_LIST

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
    iGroup.add_argument('--benchmark', type=str, default=None, metavar='TSV',
                        help='TSV file containing true per-site omega values for benchmarking')

    mGroup = parser.add_argument_group('Model options')
    mGroup.add_argument('--pi', type=str, default=None,
                        help='Pi equilibrium vector (default all equal)')
    mGroup.add_argument('--domains', type=str, default=None,
                        help='UniProt JSON file with domain annotations for hierarchical regression on omega')
    mGroup.add_argument('--reference', type=str, default=None,
                        help='Reference protein FASTA for mapping domain positions to alignment columns (required with --domains)')
    mGroup.add_argument('--regression-weight', type=float, default=0.1,
                        help='Weight of the domain regression term relative to the data likelihood (default 0.1). '
                             'Decrease to reduce influence on strong selection signals.')
    mGroup.add_argument('--only-colour-domains', action='store_true', default=False,
                        help='Run the standard model (no regression) and produce a plot coloured by domain annotation. '
                             'Requires --domains and --reference.')
    mGroup.add_argument('--estimate-uncertainty', action='store_true', default=False,
                        help='Compute per-parameter standard errors via diagonal Laplace approximation '
                             '(Hessian-based). Can be memory-intensive for large alignments.')
    mGroup.add_argument('--fit-replicates', type=int, default=1, metavar='N',
                        help='Run the optimiser N times with random perturbations of the starting values '
                             'and produce a convergence plot. Best replicate (highest log-likelihood) is used '
                             'for all downstream outputs (default: 1).')
    mGroup.add_argument('--exclude-invariant', action='store_true', default=False,
                        help='Exclude invariant sites from the GTR/theta loss and stop omega gradients '
                             'at those sites. By default invariant sites are included and the prior '
                             'regularises their omega estimates.')
    mGroup.add_argument('--output-jax', type=str, default=None, metavar='STEM',
                        help='Save MAP estimates or NUTS outputs to CSV. Default: do not save.')
    mGroup.add_argument('--fit-method', choices=['map', 'nuts'], default='map',
                        help='Fit with MAP optimisation or BlackJAX NUTS sampling (default: map).')
    mGroup.add_argument('--objective-aggregate', choices=['mean', 'sum'], default='mean',
                        help='Aggregate site log-likelihoods by mean or sum (default: mean).')
    mGroup.add_argument('--prior-mode',
                        choices=['current', 'none', 'stan_constrained', 'stan_unconstrained'],
                        default='stan_unconstrained',
                        help='Prior/Jacobian convention (default: stan_unconstrained).')
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

    sGroup = parser.add_argument_group('Sampling options')
    sGroup.add_argument('--sample-it', type=int, default=500,
                        help='Sampling iterations')

    hGroup = parser.add_argument_group('Hardware options')
    sGroup.add_argument('--platform', choices=['cpu', 'gpu'], default='cpu',
                        help='JAX platform to use (default: cpu; gpu requires an accelerator install)')
    sGroup.add_argument('--cpus', type=int, default=8,
                        help='Number of CPU cores to use')
    sGroup.add_argument('--nuts-chain-mode', choices=['sequential', 'pmap'], default='sequential',
                        help='Run NUTS chains sequentially or in parallel across JAX devices '
                             '(default: sequential).')

    other = parser.add_argument_group('Other options')
    other.add_argument('--version', action='version',
                       version='%(prog)s '+__version__)

    args = parser.parse_args()
    return args


def configure_jax_for_options(options):
    """Set JAX process flags that must exist before JAX is imported."""
    configure_platform(
        options.platform,
        cpus=options.cpus,
        force_cpu_devices=(
            options.fit_method == "nuts"
            and options.nuts_chain_mode == "pmap"
        ),
    )

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


def load_benchmark_omegas(file_name, n_sites):
    """Read true per-site omega values from a benchmark TSV file."""
    with open(file_name, newline="") as benchmark:
        reader = csv.DictReader(benchmark, delimiter="\t")
        required = {"site", "omega"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Benchmark TSV must contain 'site' and 'omega' columns")

        sites = []
        omegas = []
        for row_number, row in enumerate(reader, start=2):
            try:
                sites.append(int(row["site"]))
                omega_value = row["omega"].strip()
                omegas.append(
                    np.nan if omega_value.upper() == "NA" else float(omega_value)
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid benchmark value on TSV row {row_number}"
                ) from exc

    expected_sites = list(range(1, n_sites + 1))
    if sites != expected_sites:
        raise ValueError(
            "Benchmark TSV site values must be consecutive alignment sites "
            f"1 through {n_sites}"
        )
    return np.asarray(omegas, dtype=float)


def _most_common_amino_acid_proportions(proportions):
    """Return the most-common amino-acid proportion for each alignment site."""
    proportions = np.asarray(proportions, dtype=float)
    if proportions.ndim != 2 or proportions.shape[0] != len(CODON_LIST):
        raise ValueError("proportions must have shape (61, n_sites)")

    amino_acids = sorted(set(AA_LIST))
    amino_acid_proportions = np.zeros((len(amino_acids), proportions.shape[1]))
    amino_acid_to_row = {amino_acid: i for i, amino_acid in enumerate(amino_acids)}
    for codon_row, amino_acid in enumerate(AA_LIST):
        amino_acid_proportions[amino_acid_to_row[amino_acid]] += proportions[codon_row]
    return amino_acid_proportions.max(axis=0)


def plot_codon_proportions(X, n_samples, output_stem):
    """Save per-site codon proportions as stacked and summary plots.

    ``X`` is the 61-by-site count matrix returned by :func:`count_codons`.
    The stacked-bar output is written to ``{output_stem}_codon_proportions.pdf``.
    A line plot of the most common codon and amino acid proportions at each
    site is written to ``{output_stem}_most_common_codon_proportions.pdf``.
    """
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[0] != len(CODON_LIST):
        raise ValueError("X must have shape (61, n_sites)")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    proportions = X.astype(float) / float(n_samples)
    n_sites = X.shape[1]
    sites = np.arange(1, n_sites + 1)
    colours = plt.cm.nipy_spectral(np.linspace(0.02, 0.98, len(CODON_LIST)))
    fig_width = max(14.0, min(36.0, n_sites / 10.0))
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    bottom = np.zeros(n_sites, dtype=float)

    for codon, values, colour in zip(CODON_LIST, proportions, colours):
        ax.bar(sites, values, bottom=bottom, width=1.0, color=colour,
               edgecolor="none", label=codon)
        bottom += values

    ax.set_xlabel("Alignment codon position")
    ax.set_ylabel("Codon proportion")
    ax.set_title("Codon proportions per alignment position")
    ax.set_xlim(0.5, n_sites + 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=3,
              fontsize=7, title="Codon")
    fig.tight_layout()

    output_path = output_stem + "_codon_proportions.pdf"
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved codon proportion plot to: %s", output_path)

    most_common_proportions = proportions.max(axis=0)
    most_common_amino_acid_proportions = _most_common_amino_acid_proportions(proportions)

    summary_fig, summary_ax = plt.subplots(figsize=(fig_width, 4))
    summary_ax.plot(
        sites,
        most_common_proportions,
        linewidth=1.5,
        label="Most common codon",
    )
    summary_ax.plot(
        sites,
        most_common_amino_acid_proportions,
        linewidth=1.5,
        label="Most common amino acid",
    )
    summary_ax.set_xlabel("Alignment codon position")
    summary_ax.set_ylabel("Proportion")
    summary_ax.set_title("Most common codon and amino acid proportions per alignment position")
    summary_ax.set_xlim(0.5, n_sites + 0.5)
    summary_ax.set_ylim(0.0, 1.0)
    summary_ax.legend()
    summary_fig.tight_layout()

    summary_output_path = output_stem + "_most_common_codon_proportions.pdf"
    summary_fig.savefig(summary_output_path, format="pdf", bbox_inches="tight")
    plt.close(summary_fig)
    logging.info("Saved most common codon proportion plot to: %s", summary_output_path)
    return proportions

def main():
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True)

    options = get_options()
    configure_jax_for_options(options)

    from .sample import run_sampler

    logging.info("Reading alignment...")
    X, n_samples = count_codons(options.alignment)
    logging.info(f"Read {n_samples} samples and {X.shape[1]} codons")
    benchmark_omega = None
    if options.benchmark is not None:
        benchmark_omega = load_benchmark_omegas(options.benchmark, X.shape[1])
    if options.output_jax is not None:
        plot_codon_proportions(X, n_samples, options.output_jax)

    #print("X",X.max())
    if options.pi is None:
        pi = np.array([1/61 for i in range(61)])

    is_extracellular = None
    is_imputed = None
    regression_mask = None
    if options.domains is not None:
        if options.reference is None:
            raise ValueError("--reference is required when --domains is specified")
        logging.info("Parsing domain annotations...")
        is_extracellular, is_imputed, regression_mask = parse_domain_json(
            options.domains, options.alignment, options.reference, X.shape[1]
        )

    run_sampler(X, pi, options.sample_it, options.platform, options.cpus,
                is_extracellular, is_imputed, regression_mask, options.regression_weight,
                only_colour_domains=options.only_colour_domains,
                estimate_uncertainty=options.estimate_uncertainty,
                fit_replicates=options.fit_replicates,
                include_invariant=not options.exclude_invariant,
                output=options.output_jax,
                benchmark_omega=benchmark_omega,
                aggregate=options.objective_aggregate,
                prior_mode=options.prior_mode,
                estimate_eta=not options.fix_eta,
                fit_method=options.fit_method,
                fit_until_convergence=options.fit_until_convergence,
                convergence_tol=options.convergence_tol,
                convergence_patience=options.convergence_patience,
                convergence_check_every=options.convergence_check_every,
                convergence_min_steps=options.convergence_min_steps,
                num_warmup=options.num_warmup,
                num_samples=options.num_samples,
                num_chains=options.num_chains,
                rng_seed=options.rng_seed,
                target_acceptance_rate=options.target_acceptance_rate,
                nuts_chain_mode=options.nuts_chain_mode)

if __name__ == "__main__":
    main()
