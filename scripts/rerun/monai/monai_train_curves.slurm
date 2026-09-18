#!/bin/bash
#SBATCH --job-name=monai_curves
#SBATCH --account=gayed-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=/u/sl257/shared_data/result/scripts/rerun/logs/monai_curves_%j.out
#SBATCH --error=/u/sl257/shared_data/result/scripts/rerun/logs/monai_curves_%j.err
set -euo pipefail
source /u/sl257/miniforge3/etc/profile.d/conda.sh
conda activate pytorch_env            # monai 1.5.1 installed here (2026-09-17)
cd /u/sl257/shared_data/result
mkdir -p scripts/rerun/logs
nvidia-smi -L
python scripts/rerun/monai/prepare_masks.py
python scripts/rerun/monai/monai_train_curves.py --epochs 100 --batch-size 4 --seed 42 --num-workers 8
python scripts/compute_mask_metrics.py monai
python scripts/collect_results.py
python plot/make_figures.py
