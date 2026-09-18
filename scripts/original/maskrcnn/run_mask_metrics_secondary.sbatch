#!/bin/bash
#SBATCH --job-name=mask_metrics
#SBATCH --account=campusclusterusers
#SBATCH --partition=secondary
#SBATCH --gres=gpu:V100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/illinois/cimed/bts/gayed/data/train_script/maskrcnn_project/slurm_%j_mask_metrics.out
#SBATCH --error=/projects/illinois/cimed/bts/gayed/data/train_script/maskrcnn_project/slurm_%j_mask_metrics.err

set -euo pipefail

module load anaconda3/2024.10
module load cuda/12.4
source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate tors_env

PROJ=/projects/illinois/cimed/bts/gayed
cd "$PROJ"

python data/train_script/maskrcnn_project/eval_saved_mask_metrics.py \
  --model-path data/train_script/maskrcnn_project/run_output/maskrcnn_resnet50_fpn_best.pth \
  --output-dir data/train_script/maskrcnn_project/run_output/mask_metrics \
  --splits val test
