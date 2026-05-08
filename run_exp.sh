#!/bin/bash
#SBATCH --job-name=memory
#SBATCH --nodes=1
#SBATCH --partition=job_a100
#SBATCH --ntasks-per-node=1
# If sbatch says "configuration is not available", the GRES name may be wrong.
# Try: --gres=gpu:1   or   --gpus-per-node=1   (see: sinfo -p job_a100 -o "%P %G %c %m")
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=memory.out
#SBATCH --error=memory.err


cd /home/dennis/ok1

source /home/dennis/ok1/.venv/bin/activate


python3 inclass_simulation.py --dataset cifar10 --class-id 3 --no-pca \
  --M-min 10 --M-max 200 --mem-R 2 --n-trials 5 --device cuda

python3 inclass_simulation.py --dataset cifar10 --class-id 5 --no-pca \
  --M-min 10 --M-max       400 --mem-R 2 --n-trials 5 --device cuda

python3 inclass_simulation.py --dataset cifar10 --class-id 6 --no-pca \
  --M-min 10 --M-max       400 --mem-R 2 --n-trials 5 --device cuda

python3 inclass_simulation.py --dataset cifar10 --class-id 7 --no-pca \
  --M-min 10 --M-max       400 --mem-R 2 --n-trials 5 --device cuda

python3 inclass_simulation.py --dataset cifar10 --class-id 9 --no-pca \
  --M-min 10 --M-max       400 --mem-R 2 --n-trials 5 --device cuda



python3 inclass_simulation.py --dataset mnist --class-id 0 --no-pca \
  --M-min 10 --M-max 200 --mem-R 2 --n-trials 5 --device cuda

python3 inclass_simulation.py --dataset mnist --class-id 1 --no-pca \
  --M-min 10 --M-max 200 --mem-R 2 --n-trials 5 --device cuda

python3 inclass_simulation.py --dataset mnist --class-id 2 --no-pca \
  --M-min 10 --M-max 200 --mem-R 2 --n-trials 5 --device cuda

python3 inclass_simulation.py --dataset mnist --class-id 3 --no-pca \
--M-min 10 --M-max 200 --mem-R 2 --n-trials 5 --device cuda

python3 inclass_simulation.py --dataset mnist --class-id 9 --no-pca \
  --M-min 10 --M-max 200 --mem-R 2 --n-trials 5 --device cuda

python3 inclass_simulation.py --dataset mnist --class-id 6 --no-pca \
  --M-min 10 --M-max 200 --mem-R 2 --n-trials 5 --device cuda