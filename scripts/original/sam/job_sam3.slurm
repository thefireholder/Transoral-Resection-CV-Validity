#!/bin/bash
#SBATCH --mem=40g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=gayed-ic
#SBATCH --job-name=sam3_eval
#SBATCH --output=./logs/sam3_eval_%j.out
#SBATCH --error=./logs/sam3_eval_%j.err
#SBATCH --time=1:00:00
#SBATCH --gpus-per-node=1

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sl257@illinois.edu

source /u/sl257/miniforge3/etc/profile.d/conda.sh
conda activate pytorch_env
export HF_HOME="/u/sl257/scratch/huggingface"
export HF_HUB_OFFLINE=1

cd /u/sl257/shared_data/train_script/SAM2_project/sam2

which python3
pwd
nvidia-smi -L

python sam3_model.py 