#!/usr/bin/env python

import logging
import gzip
import numpy as np
import jax.numpy as jnp

from .__init__ import __version__
from .sample import run_sampler
from .domains import parse_domain_json

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
    mGroup.add_argument('--domains', type=str, default=None,
                        help='UniProt JSON file with domain annotations for hierarchical regression on omega')
    mGroup.add_argument('--reference', type=str, default=None,
                        help='Reference protein FASTA for mapping domain positions to alignment columns (required with --domains)')
    mGroup.add_argument('--regression-weight', type=float, default=0.1,
                        help='Weight of the domain regression term relative to the data likelihood (default 0.1). '
                             'Decrease to reduce influence on strong selection signals.')

    sGroup = parser.add_argument_group('Sampling options')
    sGroup.add_argument('--sample-it', type=int, default=500,
                        help='Sampling iterations')
    sGroup.add_argument('--warmup-it', type=int, default=500,
                        help='Warmup iterations')

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

    run_sampler(X, pi, options.warmup_it, options.sample_it, options.platform, options.cpus,
                is_extracellular, is_imputed, regression_mask, options.regression_weight)

if __name__ == "__main__":
    main()
