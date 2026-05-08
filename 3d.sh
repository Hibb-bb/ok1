#!/bin/bash

#SBATCH --job-name=memory
#SBATCH --partition=job
#SBATCH --time=24:00:00        # must be <= 02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --output=memory.out


cd /home/dennis/ok1

source /home/dennis/ok1/.venv/bin/activate

python3 capacity_simulation.py --M-min 10 --M-max 200 --n-trials 20 --max-steps 10 --d 3 --mem-R 1
python3 capacity_simulation.py --M-min 10 --M-max 200 --n-trials 20 --max-steps 10 --d 3 --mem-R 2
python3 capacity_simulation.py --M-min 10 --M-max 200 --n-trials 20 --max-steps 10 --d 3 --mem-R 3
python3 capacity_simulation.py --M-min 10 --M-max 200 --n-trials 20 --max-steps 10 --d 3 --mem-R 4
python3 capacity_simulation.py --M-min 10 --M-max 200 --n-trials 20 --max-steps 10 --d 3 --mem-R 5
python3 capacity_simulation.py --M-min 10 --M-max 200 --n-trials 20 --max-steps 10 --d 3 --mem-R 6