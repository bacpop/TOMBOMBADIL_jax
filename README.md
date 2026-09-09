<img src="https://github.com/bacpop/TOMBOMBADIL_jax/blob/main/TOMBOMBADIL_logo.png" alt="" width="200"/>

# TOMBOMBADIL
**T**ree-free **O**mega **M**apping **B**y **O**bserving **M**utations of **B**ases and **A**mino acids **D**istributed **I**nside **L**oci 

>    "Old Tom Bombadil is a merry fellow! Bright Blue his jacket is, and his boots are yellow!"
    —Tom Bombadil 

# TOMBOMBADIL - method for estimating dN/dS directly from alignments

Original implementation in Stan https://github.com/bacpop/TOMBOMBADIL

Work is based on Genomegamap https://doi.org/10.1093/molbev/msaa069

# Fitting dN/dS model to data   
Create codon-based multiple sequence alignments

install Python 3.14.0

run using  

python -m tombombadil --alignment alignment.fas.aln --fit-replicates 4 --fit-until-convergence --output-jax output.txt

The default fits one scalar omega for the complete alignment. To fit one omega
per codon site, select the per-site model explicitly:

python -m tombombadil --alignment alignment.fas.aln --omega-mode per-site --output-jax output

Optional domain JSON annotations can colour per-site omega plots. A reference
protein FASTA is required for mapping alignment columns to protein positions:

python -m tombombadil --alignment alignment.fas.aln --omega-mode per-site --domains domains.json --reference reference.faa

optional: blackJax

python -m tombombadil --alignment alignment.fas.aln --fit-method nuts --num-warmup 250 --num-samples 500 --num-chains 4 --output-jax output.txt

optional: blackJax with parallel chains

python -m tombombadil --alignment alignment.fas.aln --fit-method nuts --num-warmup 250 --num-samples 500 --num-chains 4 --nuts-chain-mode pmap --output-jax output.txt

# More options
--convergence-tol x default=1e-6

--sample-it x number of sampling steps

--platform gpu/cpu/tpu

--cpus x default=8 number of cpus

--pi uniform/empirical/F3x4 default=uniform codon equilibrium frequencies

--pi-pseudocount x default=0.5 pseudocount for empirical and F3x4 codon frequencies
