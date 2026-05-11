
### Environment

```bash
uv init
uv venv .venv
source .venv/bin/activate
uv sync
```

Activate the venv (e.g. `source .venv/bin/activate`) or prefix commands with `.venv/bin/python`.

### Capacity simulation

Image runs write under `outputs/<dataset>/<pixels|pca{d}>/Radius<R>/` (not `dim{d}/`), so PCA and raw-pixel experiments do not overwrite each other.

**Flags**

- `--no-pca` — MNIST/CIFAR use full flattened pixels (784 / 3072) after `--mem-R` scaling; no sklearn PCA.
- `--device cpu|cuda` — Euclidean MHN/DAM use PyTorch on this device. Hyperbolic image runs can use CUDA memory tensors when `GEOMSTATS_BACKEND=pytorch` (see below).
- `--no-batch` — disables batched Euclidean MHN/DAM and batched **Karcher-Flow** on CPU (falls back to scalar loops). On GPU, Karcher-Flow is always per-query.

**Hyperbolic / geomstats:** for CUDA hyperbolic dynamics, either export `GEOMSTATS_BACKEND=pytorch` or pass `--device cuda`; `capacity_simulation.py` and `inclass_simulation.py` call `recall_config.early_set_geomstats_backend_from_argv()` so a clean `python ... --device cuda ...` run sets the backend when possible. **Batched Karcher-Flow** (one dynamics step over all queries) runs only with **CPU** hyperbolic paths (`--device cpu` or synthetic, which always uses CPU for hyperboloid math). With `--device cuda`, identity Karcher uses the legacy per-query loop.

**Example commands**

```bash
# Synthetic (default): capacity curve, small trial count
python capacity_simulation.py --M-min 20 --M-max 200 --M-step 20 --n-trials 10 --max-steps 5

# MNIST + PCA; Euclidean models on GPU if available
python capacity_simulation.py --dataset mnist --device cuda --no-pca \
  --M-min 10 --M-max 1000 --n-trials 5 --max-steps 5 --mem-R 3

# CIFAR-10, raw pixels (heavy)
python capacity_simulation.py --dataset cifar10 --no-pca --device cuda \
  --M-min 10 --M-max 100 --n-trials 5 --max-steps 5 --mem-R 3

# Debug: scalar Euclidean + scalar Karcher (CPU)
python capacity_simulation.py --dataset mnist --pca-dim 50 --device cpu --no-batch \
  --M-min 10 --M-max 100 --n-trials 2 --max-steps 5
```

**Image norms (optional figure)**

```bash
python sample_image_memory.py --dataset cifar10 --pca-dim 100 --M 5000 --R 3
python sample_image_memory.py --dataset cifar10 --pca-dim 100 --M 5000 -o figures/cifar_norms.png
```

### In-class recall

Same `--no-pca`, `--device`, `--no-batch` as above. Outputs: `outputs/<dataset>/inclass/class<id>/<pixels|pca{d}>/Radius<R>/`.

```bash
python inclass_simulation.py --dataset cifar10 --class-id 3 --pca-dim 50 \
  --M-min 10 --M-max       400 --mem-R 2 --n-trials 5 --device cpu

python3 inclass_simulation.py --dataset mnist --class-id 0 --no-pca \
  --M-min 10 --M-max 200 --mem-R 3 --n-trials 5 --device cuda

python3 inclass_simulation.py --dataset cifar10 --class-id 0 --no-pca \
  --M-min 10 --M-max 200 --mem-R 3 --n-trials 5 --device cuda
```

### Tests

```bash
pytest tests/test_batch_recall_parity.py -v
```

### Cluster (Slurm) snippets

```bash
salloc -p debug_a100 -N 1 -n 1 -c 8 --mem=40G -t 02:00:00
srun --jobid=3172 --pty bash
squeue -j <JOBID> -o "%.18i %.9P %.8T %.10M %.6D %R"
```


salloc --account=p32593 --job-name=ok --nodes=1 --partition=gengpu --gres=gpu:a100:1 --ntasks-per-node=1 --cpus-per-task=16 --mem=80G --time=01:00:00


p32234

srun --jobid=7501129 --pty bash -l



python capacity_simulation.py --dataset cifar10 --device cuda --no-pca \
  --M-min 10 --M-max 1000 --n-trials 5 --max-steps 5 --mem-R 2 \
  --wandb --wandb-group "test" \
  --wandb-tags "sim:capacity,dataset:cifar10,feat:pixels,R:2,device:cuda"