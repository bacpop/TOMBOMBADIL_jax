# TOMBOMBADIL
**T**ree-free **O**mega **M**apping **B**y **O**bserving **M**utations of **B**ases and **A**mino acids **D**istributed **I**nside **L**oci

![](https://github.com/bacpop/TOMBOMBADIL_jax/blob/main/TOMBOMBADIL_logo.png)

>    "Old Tom Bombadil is a merry fellow! Bright Blue his jacket is, and his boots are yellow!"
    —Tom Bombadil

# TOMBOMBADIL - method for estimating dN/dS directly from alignments

Original implementation in Stan https://github.com/bacpop/TOMBOMBADIL

Work is based on Genomegamap https://doi.org/10.1093/molbev/msaa069

# Fitting dN/dS model to data   
run using  
poetry install  
poetry run python tombombadil-runner.py --alignment porB3.carriage.noindels.txt
