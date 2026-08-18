<img src="https://github.com/bacpop/TOMBOMBADIL_jax/blob/main/TOMBOMBADIL_logo.png" alt="" width="200"/>

# TOMBOMBADIL
**T**ree-free **O**mega **M**apping **B**y **O**bserving **M**utations of **B**ases and **A**mino acids **D**istributed **I**nside **L**oci 

>    "Old Tom Bombadil is a merry fellow! Bright Blue his jacket is, and his boots are yellow!"
    —Tom Bombadil 

# TOMBOMBADIL - method for estimating dN/dS directly from alignments

Original implementation in Stan https://github.com/bacpop/TOMBOMBADIL

Work is based on Genomegamap https://doi.org/10.1093/molbev/msaa069

# Fitting dN/dS model to data   
run using  
poetry install  
poetry run python tombombadil-runner.py --alignment porB3.carriage.noindels.txt

BlackJAX NUTS is available through `--fit-method nuts`:

```bash
python -m tombombadil --alignment alignment.fas.aln --fit-method nuts \
  --num-warmup 250 --num-samples 500 --num-chains 4 \
  --nuts-chain-mode sequential --output-jax output
```

On CPU, `--nuts-chain-mode pmap` requires JAX to see one host device per chain,
so set `--cpus` before importing JAX.

## CPU and GPU execution

CPU execution is the default and remains available with the standard install:

```bash
poetry install
poetry run python -m tombombadil --platform cpu \
  --alignment porB3.carriage.noindels.txt
```

For an NVIDIA GPU, install the accelerator-enabled JAX extra in the same
environment. CUDA 12 is the supported project baseline; choose CUDA 13 only
when the host driver and deployment environment require it:

```bash
poetry install
poetry run pip install -r requirements-gpu-cuda12.txt
poetry run python -m tombombadil --platform gpu \
  --alignment porB3.carriage.noindels.txt
```

The JAX CUDA wheels are intended for supported Linux NVIDIA environments and
include the matching CUDA runtime libraries. The NVIDIA driver must still be
installed on the host. Verify the installation with:

```bash
poetry run python -c "import jax; print(jax.devices())"
```

`--platform gpu` is strict and reports an actionable error if no GPU backend
or device is available; it does not silently fall back to CPU. `--platform
cpu` explicitly selects CPU execution even on a GPU-capable machine. For NUTS
`pmap`, CPU chains use virtual host devices controlled by `--cpus`, while GPU
chains use the visible physical GPU devices. The number of `pmap` chains may
not exceed the number of visible devices.

The model uses JAX-native operations and keeps the existing float64 setting.
GPU runs may therefore use more memory and may differ from CPU by small
floating-point or eigendecomposition tolerances.
