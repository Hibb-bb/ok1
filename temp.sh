# python3 inclass_simulation.py --dataset cifar10 --class-id 3 --no-pca \
#   --M-min 10 --M-max 200 --mem-R 2 --n-trials 5 --device cuda

# python3 inclass_simulation.py --dataset cifar10 --class-id 5 --no-pca \
#   --M-min 10 --M-max       400 --mem-R 2 --n-trials 5 --device cuda

# python3 inclass_simulation.py --dataset cifar10 --class-id 6 --no-pca \
#   --M-min 10 --M-max       400 --mem-R 2 --n-trials 5 --device cuda

# python3 inclass_simulation.py --dataset cifar10 --class-id 7 --no-pca \
#   --M-min 10 --M-max       400 --mem-R 2 --n-trials 5 --device cuda

# python3 inclass_simulation.py --dataset cifar10 --class-id 9 --no-pca \
#   --M-min 10 --M-max       400 --mem-R 2 --n-trials 5 --device cuda

python capacity_simulation.py --dataset mnist --device cuda --no-pca \
  --M-min 10 --M-max 1000 --n-trials 5 --max-steps 5 --mem-R 2

python3 inclass_simulation.py --dataset mnist --class-id 0 --no-pca \
  --M-min 10 --M-max 400 --mem-R 2 --n-trials 5 --device cuda

python3 inclass_simulation.py --dataset mnist --class-id 1 --no-pca \
  --M-min 10 --M-max 400 --mem-R 2 --n-trials 5 --device cuda

python3 inclass_simulation.py --dataset mnist --class-id 2 --no-pca \
  --M-min 10 --M-max 400 --mem-R 2 --n-trials 5 --device cuda

python3 inclass_simulation.py --dataset mnist --class-id 3 --no-pca \
--M-min 10 --M-max 200 --mem-R 2 --n-trials 5 --device cuda

python3 inclass_simulation.py --dataset mnist --class-id 9 --no-pca \
  --M-min 10 --M-max 200 --mem-R 2 --n-trials 5 --device cuda

python3 inclass_simulation.py --dataset mnist --class-id 6 --no-pca \
  --M-min 10 --M-max 200 --mem-R 2 --n-trials 5 --device cuda