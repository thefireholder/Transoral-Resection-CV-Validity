#!/bin/bash
# Dataset-size sweep, one job per (model, size) via a SLURM array.
#   python scripts/rerun/dataset_size_sweep/make_subsets.py     # once, CPU
#   sbatch --array=0-9 scripts/rerun/dataset_size_sweep/sweep.slurm
# Array index -> (model, n): 0-4 = yolo n75,150,300,600,732 ; 5-9 = maskrcnn same sizes.
# (900 and 1200 collapse to 732 = the whole training split, see make_subsets.py.)
#SBATCH --job-name=size_sweep
#SBATCH --account=gayed-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=/u/sl257/shared_data/result/scripts/rerun/logs/sweep_%A_%a.out
#SBATCH --error=/u/sl257/shared_data/result/scripts/rerun/logs/sweep_%A_%a.err
set -euo pipefail
source /u/sl257/miniforge3/etc/profile.d/conda.sh
conda activate pytorch_env
mkdir -p /u/sl257/shared_data/result/scripts/rerun/logs
SIZES=(75 150 300 600 732)
i=${SLURM_ARRAY_TASK_ID:-0}
N=${SIZES[$((i % 5))]}
SWEEP=/u/sl257/shared_data/result/scripts/rerun/dataset_size_sweep
if [ "$i" -lt 5 ]; then
  cd /u/sl257/shared_data/train_script/yolov12_project      # yolo11n-seg.pt lives here
  python $SWEEP/yolo_sweep.py --n $N --seed 0
else
  cd /u/sl257/shared_data/result
  python $SWEEP/maskrcnn_sweep.py --n $N --seed 42
fi
cd /u/sl257/shared_data/result && python plot/make_figures.py 5
