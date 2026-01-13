#!/bin/bash

#SBATCH --job-name=memory
#SBATCH --partition=job
#SBATCH --time=01:00:00        # must be <= 02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --output=memory.out


cd /home/dennis/ok1

source /home/dennis/ok1/.venv/bin/activate

python3 capacity_simulation.py --dataset mnist --pca-dim 10 --M-min 10 --M-max 1000 --n-trials 5 --max-steps 5

python capacity_simulation.py --dataset cifar10 --pca-dim 10 --M-min 10 --M-max 1000 --n-trials 5 --max-steps 5

python3 capacity_simulation.py --dataset mnist --pca-dim 20 --M-min 10 --M-max 1000 --n-trials 5 --max-steps 5

python capacity_simulation.py --dataset cifar10 --pca-dim 20 --M-min 10 --M-max 1000 --n-trials 5 --max-steps 5

# python3 capacity_simulation.py --M-min 10 --M-max 1000 --n-trials 10 --max-steps 10 --d 100 --mem-R 2

# python3 capacity_simulation.py --M-min 10 --M-max 1000 --n-trials 10 --max-steps 10 --d 200 --mem-R 2

# python3 capacity_simulation.py --M-min 10 --M-max 2000 --n-trials 10 --max-steps 10 --d 30 --mem-R 2

# python3 capacity_simulation.py --M-min 10 --M-max 2000 --n-trials 10 --max-steps 10 --d 50 --mem-R 2


# python capacity_simulation.py --dataset mnist --pca-dim 100 --M-min 10 --M-max 500 

# python capacity_simulation.py --dataset cifar10 --pca-dim 100 --M-min 10 --M-max 500

# python capacity_simulation.py --dataset mnist --pca-dim 200 --M-min 10 --M-max 500 

# python capacity_simulation.py --dataset cifar10 --pca-dim 200 --M-min 10 --M-max 500

# python capacity_simulation.py --dataset mnist --pca-dim 30 --M-min 10 --M-max 500

# python capacity_simulation.py --dataset cifar10 --pca-dim 30 --M-min 10 --M-max 500