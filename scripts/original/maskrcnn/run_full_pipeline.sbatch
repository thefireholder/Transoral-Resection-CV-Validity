#!/bin/bash
#SBATCH --job-name=maskrcnn_full
#SBATCH --account=campusclusterusers
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:V100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/projects/illinois/cimed/bts/gayed/data/train_script/maskrcnn_project/slurm_%j_full.out
#SBATCH --error=/projects/illinois/cimed/bts/gayed/data/train_script/maskrcnn_project/slurm_%j_full.err

set -euo pipefail

module load anaconda3/2024.10
module load cuda/12.4
source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate tors_env

PROJ=/projects/illinois/cimed/bts/gayed
cd "$PROJ"
mkdir -p data/train_script/maskrcnn_project/run_output/curves

echo "=========================================="
echo " STEP 1/2: Train + Evaluate"
echo "=========================================="
python data/train_script/maskrcnn_project/train_maskrcnn.py \
  --epochs 5 \
  --batch-size 4 \
  --lr 0.005 \
  --num-workers 4 \
  --output-dir data/train_script/maskrcnn_project/run_output

echo "=========================================="
echo " STEP 2/2: Generate Curves (saves raw predictions JSON)"
echo "=========================================="
python data/train_script/maskrcnn_project/generate_curves.py \
  --splits train val test \
  --iou-threshold 0.5

echo "=========================================="
echo " Pipeline complete."
echo " Outputs:"
echo "   Best checkpoint : run_output/maskrcnn_resnet50_fpn_best.pth"
echo "   Final checkpoint: run_output/maskrcnn_resnet50_fpn_final.pth"
echo "   Metrics JSON    : run_output/maskrcnn_eval_metrics.json"
echo "   Raw predictions : run_output/curves/raw_predictions_{train,val,test}.json"
echo "   Curve CSVs      : run_output/curves/{train,val,test}_curve.csv"
echo "   Curve PNGs      : run_output/curves/*.png"
echo "=========================================="
