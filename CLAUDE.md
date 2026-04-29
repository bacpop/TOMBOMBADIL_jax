# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Folder Names
- Target folder: TOMMBOMBADIL_jax
- Absolute path: /Users/llorenz/Documents/PhD_Project/Code/SelectionModel/TOMBOMBADIL_jax_vibecoding/TOMBOMBADIL_jax

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

## Planned Feature: Domain-Informed Hierarchical Regression on Omega

### Goal
Co-estimate a hierarchical regression on omega alongside all other model parameters. The regression uses external protein domain annotations (extracellular vs. other) to encourage omega estimates for sites in the same domain type to be more similar to each other.

### Statistical model
The regression applies to omega in log space:

```
log(omega_i) = α + β × is_extracellular_i + ε_i
ε_i ~ Normal(0, σ²)
```

- `α` (intercept), `β` (slope), and `σ` (residual std) are free parameters, co-estimated with all other model parameters via gradient descent
- `is_extracellular_i` is a fixed binary covariate (0 or 1) derived from the domain JSON for site i
- `ε_i` is a per-site normally distributed random effect
- The regression term is added to the log-likelihood (not as a standalone prior), so that it participates in gradient-based co-estimation
- Start with MAP: treat `ε_i` as optimised parameters rather than marginalising analytically

### New parameters
| Parameter | Description | Initial value |
|---|---|---|
| `alpha_reg` | Intercept of regression on log(omega) | 0.0 |
| `beta_reg` | Effect of being extracellular on log(omega) | 0.0 |
| `log_sigma` | Log of residual std σ (unconstrained) | 0.0 |

All three live in unconstrained space and are included in the existing parameter dict alongside the GTR parameters.

### Domain JSON parsing
Domain annotations come from UniProt JSON files (one per protein). Relevant fields:
```json
{
  "features": [
    {
      "type": "Topological domain",
      "location": {
        "start": { "value": 23, "modifier": "EXACT" },
        "end":   { "value": 62, "modifier": "EXACT" }
      },
      "description": "Extracellular"
    }
  ]
}
```

Parsing rules:
- Keep only features where `description == "Extracellular"` (case-insensitive). Ignore all other descriptions.
- Ignore the `type` field entirely.
- Positions are 1-based protein sequence positions — convert to 0-based codon indices.
- Skip features where `modifier != "EXACT"` on either start or end (warn the user).
- Output: a binary `jnp.array` of shape `(n_sites,)` named `is_extracellular` (1 = extracellular, 0 = everything else including unannotated sites).
- Domain parsing and covariate construction happen **outside** jit (fixed data, not traced).

### Integration points
- Add `alpha_reg`, `beta_reg`, `log_sigma` to the parameter dict in `run_sampler()`.
- Add a `regression_log_likelihood(params, is_extracellular)` function (in `sample.py` or a new `domains.py`) that computes the hierarchical log-likelihood contribution.
- Call this from the main loss, summing with existing terms.
- **Clarify with the user** whether `log(omega_i)` predicted by the regression should replace or inform the current omega parameterisation before changing it.
- Do **not** change GTR parameters, their parameterisation, or any existing omega parameterisation outside of adding the regression term.

### Testing requirements
- Test the parser on a UniProt JSON with known extracellular regions to verify `is_extracellular` is correct.
- Verify gradients flow through `alpha_reg`, `beta_reg`, `log_sigma` using `jax.grad`.
- Smoke test: with `beta=0` and large `sigma`, regression should have negligible effect on omega estimates.
- Sanity check: with very small `sigma`, all omega values should converge toward the regression prediction.

### Out of scope (for now)
- Domain types other than "Extracellular"
- Multi-category regression
- Fully Bayesian treatment of regression parameters (MCMC or VI over α, β, σ)
