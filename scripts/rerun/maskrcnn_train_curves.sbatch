#!/bin/bash
#SBATCH --job-name=mrcnn_curves
#SBATCH --account=gayed-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=10:00:00
#SBATCH --output=/u/sl257/shared_data/result/scripts/rerun/logs/mrcnn_curves_%j.out
#SBATCH --error=/u/sl257/shared_data/result/scripts/rerun/logs/mrcnn_curves_%j.err
set -euo pipefail
source /u/sl257/miniforge3/etc/profile.d/conda.sh
conda activate pytorch_env            # torch 2.6 + torchvision + pycocotools + cv2 -- all present
cd /u/sl257/shared_data/result
mkdir -p scripts/rerun/logs
nvidia-smi -L
python scripts/rerun/maskrcnn_train_curves.py --epochs 12 --lr-steps 8 11 --batch-size 4 --lr 0.005 --seed 42 --adopt
python scripts/compute_mask_metrics.py maskrcnn   # rescore the adopted 12-epoch model
python scripts/collect_results.py                 # picks up maskrcnn_rerun.csv + superseded flag
python plot/make_figures.py
