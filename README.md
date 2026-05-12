# Hyperbolic neural population geometry benefits computation

This repository contains the code to reproduce the simulations in "Hyperbolic neural population geometry benefits computation".

Simulation code lives in the **`icml_hyp`** package under **`src/`**. Root scripts such as **`capacity_simulation.py`** are thin wrappers; they delegate to **`icml_hyp.cli`**. Jupyter notebooks import **`icml_hyp`** directly, so the package needs to be on your Python path.

### Environment

From the **`ok1/`** directory (repository root):

```bash
uv venv
source .venv/bin/activate   # Linux/macOS; on Windows use .venv\Scripts\activate
uv sync
uv pip install -e .
```

Activate the venv or prefix commands with `.venv/bin/python`.


### Capacity simulation

Image runs write under `outputs/<dataset>/<pixels|pca{d}>/Radius<R>/beta<β>/` (e.g. `beta10`, `beta1`).

**Quick start (two examples)**


```bash
python capacity_simulation.py --M-min 20 --M-max 200 --M-step 20 --n-trials 10 --max-steps 5

python capacity_simulation.py --dataset mnist --no-pca --device cuda \
  --M-min 10 --M-max 400 --n-trials 5 --max-steps 5 --mem-R 3
```

**More example commands**

```bash
python capacity_simulation.py --dataset mnist --device cuda --pca-dim 20 \
  --M-min 10 --M-max 1000 --n-trials 5 --max-steps 5 --mem-R 3

python capacity_simulation.py --dataset cifar10 --no-pca --device cuda \
  --M-min 10 --M-max 100 --n-trials 5 --max-steps 5 --mem-R 3
```


### In-class recall

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



```bash
salloc --nodes=1 --partition=<gpu_partition> --gres=gpu:<type>:1 --cpus-per-task=16 --mem=80G --time=01:00:00
# then on the allocation:
python capacity_simulation.py --dataset cifar10 --device cuda --no-pca \
  --M-min 10 --M-max 1000 --n-trials 5 --max-steps 5 --mem-R 2 \
  --wandb --wandb-group "capacity" \
  --wandb-tags "sim:capacity,dataset:cifar10,feat:pixels,R:2,device:cuda"
```

## Cite

If you find our paper useful, please consider citing our paper:
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
