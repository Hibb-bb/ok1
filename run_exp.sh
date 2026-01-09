#!/bin/bash

#SBATCH --job-name=memory
#SBATCH --partition=job
#SBATCH --time=48:00:00        # must be <= 02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --output=memory.out


cd /home/dennis/ok1

source /home/dennis/ok1/.venv/bin/activate

# python3 capacity_simulation.py --M-min 10 --M-max 1000 --n-trials 1 --max-steps 5 --d 10

python3 capacity_simulation.py --M-min 10 --M-max 1000 --n-trials 10 --max-steps 5 --d 20

python3 capacity_simulation.py --M-min 10 --M-max 2000 --n-trials 10 --max-steps 5 --d 30

cd exp

python3 r2.py