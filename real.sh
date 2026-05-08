#!/bin/bash

#SBATCH --job-name=memory
#SBATCH --partition=job
#SBATCH --time=24:00:00        # must be <= 02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --output=memory.out


cd /home/dennis/ok1

source /home/dennis/ok1/.venv/bin/activate

# Real-world

echo "MNIST 10"
python3 capacity_simulation.py --dataset mnist --pca-dim 100 --M-min 10 --M-max 1000 --n-trials 5 --max-steps 5

echo "CIFAR 10"
python capacity_simulation.py --dataset cifar10 --pca-dim 100 --M-min 10 --M-max 1000 --n-trials 5 --max-steps 5

echo "MNIST 20"
python3 capacity_simulation.py --dataset mnist --pca-dim 50 --M-min 10 --M-max 1000 --n-trials 5 --max-steps 5

echo "CIFAR 20"
python capacity_simulation.py --dataset cifar10 --pca-dim 50 --M-min 10 --M-max 1000 --n-trials 5 --max-steps 5

echo "real world done"

# R = 2
