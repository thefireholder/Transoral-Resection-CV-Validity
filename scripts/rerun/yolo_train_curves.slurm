#!/bin/bash
#SBATCH --job-name=yolo_curves
#SBATCH --account=gayed-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=/u/sl257/shared_data/result/scripts/rerun/logs/yolo_curves_%j.out
#SBATCH --error=/u/sl257/shared_data/result/scripts/rerun/logs/yolo_curves_%j.err
set -euo pipefail
source /u/sl257/miniforge3/etc/profile.d/conda.sh
conda activate pytorch_env
cd /u/sl257/shared_data/train_script/yolov12_project     # so yolo11n-seg.pt (pretrained weights) is found
mkdir -p /u/sl257/shared_data/result/scripts/rerun/logs
nvidia-smi -L
python /u/sl257/shared_data/result/scripts/rerun/yolo_train_curves.py --epochs 60 --device 0
cd /u/sl257/shared_data/result
python scripts/collect_results.py
python plot/make_figures.py 1a
