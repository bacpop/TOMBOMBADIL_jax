# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TOMBOMBADIL (Tree-free Omega Mapping By Observing Mutations of Bases and Amino acids Distributed Inside Loci) is a JAX rewrite of a codon-level selection model. It estimates per-site dN/dS ratios (omega) from a codon alignment without requiring a phylogenetic tree.

## Commands

### Run the model
```bash
python -m tombombadil --alignment <alignment.fasta>
# or
python tombombadil-runner.py --alignment <alignment.fasta>
```

Key options: `--sample-it` (default 500), `--warmup-it` (default 500), `--platform cpu|gpu|tpu`, `--cpus`

### Run tests
```bash
# Run individual test classes:
python -m unittest -v test.test_fn.Testdiv
python -m unittest -v test.test_fn.Test_codon_count_matrix

# Run all tests:
python -m unittest discover -v test
```

Tests must be run from the repository root (the fasta test file `porB3.carriage.noindels.txt` is referenced by relative path).

### Install
```bash
# Uses poetry; requires Python >=3.14
pip install .
# or
poetry install
```

## Architecture

### Data flow
1. `count_codons()` (`__main__.py`) reads a FASTA alignment (plain or gzipped) and builds a **(61, N)** integer matrix `X` — 61 sense codons × N alignment positions. Stop codons and ambiguous bases (non-ACGT) are excluded. Codons are reordered from alphabetical (ACGT) to the biological standard (TCAG) via `col_order`.
2. `run_sampler()` (`sample.py`) orchestrates fitting: it precomputes pi-related matrix transforms, builds a mask for invariant sites, initializes parameters, and runs gradient-based optimization (optax Adam).
3. `model()` (`sample.py`) computes the log-likelihood for a single site using `gen_alpha()` → Dirichlet-Multinomial log-PMF → logsumexp over ancestral states.
4. `batched_loss` uses `jax.vmap` to map `model()` over all alignment columns simultaneously.

### Key modules
- **`gtr.py`**: Builds the 61×61 GTR substitution rate matrix. `build_GTR()` constructs from scratch; `update_GTR()` updates an existing matrix with a new omega. Uses a precomputed `omega_mat` (61×61 boolean matrix indicating nonsynonymous pairs).
- **`likelihood.py`**: `gen_alpha()` computes the transition probability matrix via eigendecomposition (`jnp.linalg.eigh`). A small jitter (1e-6) is added to the diagonal to avoid degenerate eigenvalues. The output `muti = m_AB + I` represents codon substitution frequencies used as Dirichlet concentration parameters.
- **`sample.py`**: Contains the model, parameter transforms, and optimizer. Parameters are stored in unconstrained space and mapped to positive reals via `positive(x) = exp(x) + 1e-6`. Gradient stopping is applied for: (1) sites with no diversity (`mask==0`), (2) sites where omega ≤ 0.01.

### Parameters
The model fits 8 parameters total:
- `alpha, beta, gamma, delta, epsilon, eta` — 6 GTR exchangeability rates
- `theta` — overall mutation rate scalar
- `omega` — per-site dN/dS vector of length N (one per alignment column)

### Critical JAX notes
- `jax.config.update('jax_enable_x64', True)` is set in `sample.py` — required to avoid NaN issues during optimization.
- The model is JIT-compiled via `@jit`. Debug `print` statements inside jitted functions need `jax.debug.print`.
- The `for` loops in `gen_alpha()` and `build_GTR()` unroll at trace time — avoid adding new Python loops without considering the compilation cost.
