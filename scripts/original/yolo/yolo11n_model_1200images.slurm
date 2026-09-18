#!/bin/bash
#SBATCH --mem=40g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=gayed-ic
#SBATCH --job-name=yolo11n_1200images
#SBATCH --output=./logs/yolo11n_1200images_%j.out
#SBATCH --error=./logs/yolo11n_1200images_%j.err
#SBATCH --time=4:00:00
#SBATCH --gpus-per-node=1

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sl257@illinois.edu

source /u/sl257/miniforge3/etc/profile.d/conda.sh
conda activate pytorch_env

cd /u/sl257/shared_data/train_script/yolov12_project

which python3
pwd
nvidia-smi -L

python yolo11n_model.py
