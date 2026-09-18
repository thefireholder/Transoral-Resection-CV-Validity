#!/bin/bash
#SBATCH --job-name=sam3_gtbox
#SBATCH --account=gayed-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/u/sl257/shared_data/result/scripts/rerun/logs/sam3_save_%j.out
#SBATCH --error=/u/sl257/shared_data/result/scripts/rerun/logs/sam3_save_%j.err
set -euo pipefail
source /u/sl257/miniforge3/etc/profile.d/conda.sh
conda activate pytorch_env
export HF_HOME="/u/sl257/scratch/huggingface"     # facebook/sam3 weights are cached here
export HF_HUB_OFFLINE=1
cd /u/sl257/shared_data/result
mkdir -p scripts/rerun/logs
nvidia-smi -L
python scripts/rerun/sam3/sam3_gt_box_prompt.py
python scripts/compute_mask_metrics.py sam3_gtbox
python scripts/collect_results.py
python plot/make_figures.py
