
### Environment

```uv init```
```uv venv .venv```
```uv sync```


### Capacity simulation

```
python3 capacity_simulation.py --M-min 100 --M-max 150 --n-trials 5 --max-steps 5
```



salloc -p debug -N 1 -n 1 -c 8 --mem=32G -t 02:00:00
srun --jobid=2990 --pty bash

squeue -j 2942 -o "%.18i %.9P %.8T %.10M %.6D %R"