#!/bin/bash
#SBATCH --account=p32593            # your allocation ID (e.g., p31911)
#SBATCH --partition=gengpu            # GPU partition on Quest
#SBATCH --gres=gpu:a100:1             # 1 A100 GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16          # 16 CPUs
#SBATCH --mem=80G                     # 40 GB RAM
#SBATCH --time=10:00:00               # 2 hours (hh:mm:ss)
#SBATCH --job-name=icml_vis
#SBATCH --output=icml_reproduce_vis.out      # %x = job name, %j = job ID
#SBATCH --error=icml_reproduce_vis.err
#SBATCH --mail-type=BEGIN,END,FAIL    # or ALL
#SBATCH --mail-user=hibb@u.northwestern.edu


# Use dir you submitted from (run: cd .../ok1 && sbatch run_exp.sh), or set OK1_ROOT.
if [[ -n "${OK1_ROOT:-}" ]]; then
  cd "$OK1_ROOT" || { echo "cd failed: $OK1_ROOT"; exit 1; }
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "$SLURM_SUBMIT_DIR" || { echo "cd failed: SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"; exit 1; }
else
  cd /gpfs/projects/p32593/ok1 2>/dev/null || cd /projects/p32593/ok1 || { echo "cd failed: set OK1_ROOT or submit with sbatch from ok1/"; exit 1; }
fi

source .venv/bin/activate
uv sync


python3 vis.py --N 100 --sigma 0.1 0.2 0.35 0.5 --out outputs/vis/demo.png --no-show --no-pca
