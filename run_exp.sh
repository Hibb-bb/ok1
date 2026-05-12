#!/bin/bash
#SBATCH --account=p32593            # your allocation ID (e.g., p31911)
#SBATCH --partition=gengpu            # GPU partition on Quest
#SBATCH --gres=gpu:a100:1             # 1 A100 GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16          # 16 CPUs
#SBATCH --mem=80G                     # 40 GB RAM
#SBATCH --time=15:00:00               # 2 hours (hh:mm:ss)
#SBATCH --job-name=icml
#SBATCH --output=icml_reproduce.out      # %x = job name, %j = job ID
#SBATCH --error=icml_reproduce.err
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

WANDB_GROUP="${WANDB_GROUP:-run_exp_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)}"

python3 inclass_simulation.py --dataset cifar10 --class-id 3 --no-pca --beta 100 \
  --M-min 10 --M-max 400 --mem-R 2 --n-trials 5 --device cuda \
  --wandb --wandb-group "$WANDB_GROUP" \
  --wandb-tags "sim:inclass,dataset:cifar10,class:3,feat:pixels,R:2,device:cuda"

python3 inclass_simulation.py --dataset cifar10 --class-id 5 --no-pca --beta 100 \
  --M-min 10 --M-max       400 --mem-R 2 --n-trials 5 --device cuda \
  --wandb --wandb-group "$WANDB_GROUP" \
  --wandb-tags "sim:inclass,dataset:cifar10,class:5,feat:pixels,R:2,device:cuda"

python3 inclass_simulation.py --dataset cifar10 --class-id 6 --no-pca --beta 100 \
  --M-min 10 --M-max       400 --mem-R 2 --n-trials 5 --device cuda \
  --wandb --wandb-group "$WANDB_GROUP" \
  --wandb-tags "sim:inclass,dataset:cifar10,class:6,feat:pixels,R:2,device:cuda"

python3 inclass_simulation.py --dataset cifar10 --class-id 7 --no-pca --beta 100 \
  --M-min 10 --M-max       400 --mem-R 2 --n-trials 5 --device cuda \
  --wandb --wandb-group "$WANDB_GROUP" \
  --wandb-tags "sim:inclass,dataset:cifar10,class:7,feat:pixels,R:2,device:cuda"

python3 inclass_simulation.py --dataset cifar10 --class-id 9 --no-pca --beta 100 \
  --M-min 10 --M-max       400 --mem-R 2 --n-trials 5 --device cuda \
  --wandb --wandb-group "$WANDB_GROUP" \
  --wandb-tags "sim:inclass,dataset:cifar10,class:9,feat:pixels,R:2,device:cuda"



python3 inclass_simulation.py --dataset mnist --class-id 0 --no-pca --beta 100 \
  --M-min 10 --M-max 200 --mem-R 2 --n-trials 5 --device cuda \
  --wandb --wandb-group "$WANDB_GROUP" \
  --wandb-tags "sim:inclass,dataset:mnist,class:0,feat:pixels,R:2,device:cuda"

python3 inclass_simulation.py --dataset mnist --class-id 1 --no-pca --beta 100 \
  --M-min 10 --M-max 200 --mem-R 2 --n-trials 5 --device cuda \
  --wandb --wandb-group "$WANDB_GROUP" \
  --wandb-tags "sim:inclass,dataset:mnist,class:1,feat:pixels,R:2,device:cuda"

python3 inclass_simulation.py --dataset mnist --class-id 2 --no-pca --beta 100 \
  --M-min 10 --M-max 200 --mem-R 2 --n-trials 5 --device cuda \
  --wandb --wandb-group "$WANDB_GROUP" \
  --wandb-tags "sim:inclass,dataset:mnist,class:2,feat:pixels,R:2,device:cuda"

python3 inclass_simulation.py --dataset mnist --class-id 3 --no-pca --beta 100 \
--M-min 10 --M-max 200 --mem-R 2 --n-trials 5 --device cuda \
  --wandb --wandb-group "$WANDB_GROUP" \
  --wandb-tags "sim:inclass,dataset:mnist,class:3,feat:pixels,R:2,device:cuda"

python3 inclass_simulation.py --dataset mnist --class-id 9 --no-pca --beta 100 \
  --M-min 10 --M-max 200 --mem-R 2 --n-trials 5 --device cuda \
  --wandb --wandb-group "$WANDB_GROUP" \
  --wandb-tags "sim:inclass,dataset:mnist,class:9,feat:pixels,R:2,device:cuda"

python3 inclass_simulation.py --dataset mnist --class-id 6 --no-pca --beta 100 \
  --M-min 10 --M-max 200 --mem-R 2 --n-trials 5 --device cuda \
  --wandb --wandb-group "$WANDB_GROUP" \
  --wandb-tags "sim:inclass,dataset:mnist,class:6,feat:pixels,R:2,device:cuda"



python3 capacity_simulation.py --dataset mnist --device cuda --no-pca \
  --M-min 10 --M-max 1000 --n-trials 5 --max-steps 5 --mem-R 2 \
  --wandb --wandb-group "$WANDB_GROUP" \
  --wandb-tags "sim:capacity,dataset:mnist,feat:pixels,R:2,device:cuda"

python3 capacity_simulation.py --dataset cifar10 --device cuda --no-pca \
  --M-min 10 --M-max 1000 --n-trials 5 --max-steps 5 --mem-R 2 \
  --wandb --wandb-group "$WANDB_GROUP" \
  --wandb-tags "sim:capacity,dataset:cifar10,feat:pixels,R:2,device:cuda"




python3 capacity_simulation.py --dataset mnist --device cuda --no-pca --beta 100 \
  --M-min 10 --M-max 1000 --n-trials 5 --max-steps 5 --mem-R 2 \
  --wandb --wandb-group "$WANDB_GROUP" \
  --wandb-tags "sim:capacity,dataset:mnist,feat:pixels,R:2,device:cuda"

python3 capacity_simulation.py --dataset cifar10 --device cuda --no-pca --beta 100 \
  --M-min 10 --M-max 1000 --n-trials 5 --max-steps 5 --mem-R 2 \
  --wandb --wandb-group "$WANDB_GROUP" \
  --wandb-tags "sim:capacity,dataset:cifar10,feat:pixels,R:2,device:cuda"



python3 capacity_simulation.py --dataset synthetic --device cuda \
  --M-min 10 --M-max 1000 --n-trials 5 --max-steps 5 --mem-R 1 \
  --wandb --wandb-group "$WANDB_GROUP" \
  --wandb-tags "sim:capacity,dataset:syn,feat:pixels,R:1,device:cuda"

python3 capacity_simulation.py --dataset synthetic --device cuda \
  --M-min 10 --M-max 1000 --n-trials 5 --max-steps 5 --mem-R 2 \
  --wandb --wandb-group "$WANDB_GROUP" \
  --wandb-tags "sim:capacity,dataset:syn,feat:pixels,R:2,device:cuda"

python3 capacity_simulation.py --dataset synthetic --device cuda \
  --M-min 10 --M-max 1000 --n-trials 5 --max-steps 5 --mem-R 3 \
  --wandb --wandb-group "$WANDB_GROUP" \
  --wandb-tags "sim:capacity,dataset:syn,feat:pixels,R:3,device:cuda"