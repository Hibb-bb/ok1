
<h1 align="center">Hyperbolic neural population geometry benefits computation </h1>

<p align="center">
  This repository contains code for the following paper:
</p>

<blockquote align="center">
  <b>BWLer: Hyperbolic neural population geometry benefits computation</b><br/>
  Dennis Wu, Yi-Chun Hung, Braden Yuille, James E. Fitzgerald*, Han Liu*<br/>
  <small><a href="https://openreview.net/forum?id=WXNjDNDnpy"><em>[Read the paper]</em></a></small>
</blockquote>

<div style="margin-top: 0.75em;"></div>

This paper proves that under <b>exponentially distributed place field sizes<b>, the population geometry induced by the hippocampal place cells is hyperbolic.

<p>
  This repository provides implementation for:
</p>

<ul>
  <li><b>Karcher-flow Model:</b> A computational model of associative memory that operates on the hyperboloid model. </li>
  <li><b>Karcher-flow layers:</b> Hyperbolic machine learning layers inspired by the Karcher-flow model.</li>
</ul>


<b> Add figures </b>

Simulation code lives in the **`icml_hyp`** package under **`src/`**. Root scripts such as **`capacity_simulation.py`** are thin wrappers; they delegate to **`icml_hyp.cli`**. Jupyter notebooks import **`icml_hyp`** directly, so the package needs to be on your Python path.

### Dependencies

From the **`ok1/`** directory (repository root):

```bash
uv venv
source .venv/bin/activate   # Linux/macOS; on Windows use .venv\Scripts\activate
uv sync
uv pip install -e .
```


### Code structure

The codebase is organized as follows:

- [scripts/](scripts/): contains scripts for running the experiments:
  - [scripts/capacity_simulation.py](scripts/capacity_simulation.py): capacity simulation scripts
  - [scripts/inclass_simulation.py](scripts/inclass_simulation.py): capacity simulation within similar patterns scripts
- [src/experiments/cli](src/experiments/cli): main experiment interface
- [src/experiments/recall](src/experiments/recall): implementation of the Karcher-flow model and other baselines
- [src/experiments/config](src/experiments/config): contains configs of the recall simulation


### Quick start

##### Capacity simulation

Using cuda is highly recommended for simulations

```bash
python capacity_simulation.py --M-min 20 --M-max 200 --M-step 20 --n-trials 10 --max-steps 5

python capacity_simulation.py --dataset mnist --no-pca --device cuda \
  --M-min 10 --M-max 400 --n-trials 5 --max-steps 5 --mem-R 3
```

Image runs write under `outputs/<dataset>/<pixels|pca{d}>/Radius<R>/beta<β>/` (e.g. `beta10`, `beta1`).


#####  In-class simulation

Outputs: `outputs/<dataset>/inclass/class<id>/<pixels|pca{d}>/Radius<R>/beta<β>/`.

```bash
python3 inclass_simulation.py --dataset cifar10 --class-id 3 --pca-dim 50 \
  --M-min 10 --M-max 400 --mem-R 2 --n-trials 5 --device cpu

python3 inclass_simulation.py --dataset mnist --class-id 0 --no-pca \
  --M-min 10 --M-max 200 --mem-R 3 --n-trials 5 --device cuda

python3 inclass_simulation.py --dataset cifar10 --class-id 0 --no-pca \
  --M-min 10 --M-max 200 --mem-R 3 --n-trials 5 --device cuda
```

To run all simulations, see `run_exp.sh`


### Cite

If you find this work useful, please consider citing our paper:
```bibtex
@inproceedings{
anonymous2026hyperbolic,
title={Hyperbolic neural population geometry benefits computation},
author={Anonymous},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=WXNjDNDnpy}
}
```



salloc --account=p32593 --job-name=ok --nodes=1 --partition=gengpu --gres=gpu:a100:1 --ntasks-per-node=1 --cpus-per-task=16 --mem=80G --time=02:00:00

srun --jobid=7802381 --pty bash -l
