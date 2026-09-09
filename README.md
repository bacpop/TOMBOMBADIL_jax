<img src="https://github.com/bacpop/TOMBOMBADIL_jax/blob/main/TOMBOMBADIL_logo.png" alt="" width="200"/>

# TOMBOMBADIL
**T**ree-free **O**mega **M**apping **B**y **O**bserving **M**utations of **B**ases and **A**mino acids **D**istributed **I**nside **L**oci 

>    "Old Tom Bombadil is a merry fellow! Bright Blue his jacket is, and his boots are yellow!"
    —Tom Bombadil 

# TOMBOMBADIL - method for estimating dN/dS directly from alignments

Original implementation in Stan https://github.com/bacpop/TOMBOMBADIL

Work is based on Genomegamap https://doi.org/10.1093/molbev/msaa069

# Fitting dN/dS model to data   
1. Create codon-based multiple sequence alignments

2. Install Python 3.14.0, clone this GitHub repository

3. Estimate dN/dS using TOMBOMBADIL by running one of the following commands from within the folder  

- Fit one omega estimate for the whole alignment (scalar omega) with maximum a posteriori (MAP) optimisation (default)
```bash
python -m tombombadil --alignment alignment.fas.aln --fit-replicates 4 --fit-until-convergence --output-jax output.txt
```

- Fit one omega estimate for per codon position in the alignment with maximum a posteriori (MAP) optimisation
```bash
python -m tombombadil --alignment alignment.fas.aln --omega-mode per-site --output-jax output
```

Optional domain JSON annotations can colour per-site omega plots. A reference
protein FASTA is required for mapping alignment columns to protein positions:

```bash
python -m tombombadil --alignment alignment.fas.aln --omega-mode per-site --domains domains.json --reference reference.faa
```


- Scalar omega parameter inference with MCMC (NUTS) using blackJax

```bash
python -m tombombadil --alignment alignment.fas.aln --fit-method nuts --num-warmup 250 --num-samples 500 --num-chains 4 --output-jax output.txt
```

- Scalar omega parameter inference with MCMC (NUTS) using blackJax with parallel chains

```bash
python -m tombombadil --alignment alignment.fas.aln --fit-method nuts --num-warmup 250 --num-samples 500 --num-chains 4 --nuts-chain-mode pmap --output-jax output.txt
```

- Fit one omega estimate for per codon position in the alignment with MCMC (NUTS) using blackJax 
```bash
python -m tombombadil --alignment alignment.fas.aln --omega-mode per-site --fit-method nuts --num-warmup 250 --num-samples 500 --num-chains 4 --output-jax output.txt
```


## Mini tutorial

The repository includes the example codon alignment of porin porB of *Neisseria meningitidis* `porB3_aligned.fasta`.

### 1. Scalar omega with maximum a posteriori (MAP) optimisation
(takes around 30 seconds on one cpu core)

This is the default model: one omega is estimated for the complete alignment.
`--sample-it` controls the number of MAP optimisation steps.

```bash
python -m tombombadil \
  --alignment porB3_aligned.fasta \
  --omega-mode scalar \
  --fit-method map \
  --sample-it 500 \
  --output-jax porB3_map
```

This writes `scalar_porB3_map_Allparams.csv`.

### 2. Per-site omega with maximum a posteriori (MAP) optimisation
(takes about four minutes on one cpu core)

This estimates one omega value for each codon site.

```bash
python -m tombombadil \
  --alignment porB3_aligned.fasta \
  --omega-mode per-site \
  --fit-method map \
  --sample-it 500 \
  --output-jax porB3_map
```

This writes `per_site_porB3_map_GTRparams.csv`,
`per_site_porB3_map_omega.csv`, and a per-site omega plot.

### 3. Scalar omega with NUTS sampling

NUTS estimates a posterior distribution for one alignment-wide omega.

```bash
python -m tombombadil \
  --alignment porB3_aligned.fasta \
  --omega-mode scalar \
  --fit-method nuts \
  --num-warmup 250 \
  --num-samples 500 \
  --num-chains 4 \
  --output-jax porB3_nuts
```

### 4. Per-site omega with NUTS sampling

This samples a posterior distribution for every codon-site omega. It is more
computationally demanding because the parameter dimension grows with the
alignment length.

```bash
python -m tombombadil \
  --alignment porB3_aligned.fasta \
  --omega-mode per-site \
  --fit-method nuts \
  --num-warmup 250 \
  --num-samples 500 \
  --num-chains 4 \
  --output-jax porB3_nuts
```

For CPU parallel chains, add `--nuts-chain-mode pmap --cpus 4`. Otherwise,
chains run sequentially by default. NUTS writes posterior samples and summary
files with the selected `scalar_` or `per_site_` prefix.

Output files are prefixed by omega mode. With `--output-jax output`, scalar
MAP fitting writes `scalar_output_Allparams.csv`; per-site MAP fitting writes
`per_site_output_GTRparams.csv` and `per_site_output_omega.csv`. NUTS files
use the same `scalar_` or `per_site_` prefix.

# More options
--convergence-tol x default=1e-6

--sample-it x number of sampling steps

--platform gpu/cpu/tpu

--cpus x default=1 number of cpus for CPU pmap chains

--pi uniform/empirical/F3x4 default=uniform codon equilibrium frequencies

--pi-pseudocount x default=0.5 pseudocount for empirical and F3x4 codon frequencies
