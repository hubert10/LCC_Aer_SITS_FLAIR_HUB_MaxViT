#!/bin/bash 
#SBATCH --job-name=LCC_Aer_SITS_FLAIR_HUB_MaxViT_pred
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=20:00:00
#SBATCH --mail-user=kanyamahanga@ipi.uni-hannover.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output logs/LCC_Aer_SITS_FLAIR_HUB_MaxViT_pred_%j.out
#SBATCH --error logs/LCC_Aer_SITS_FLAIR_HUB_MaxViT_pred_%j.err
export CONDA_ENVS_PATH=$HOME/.conda/envs
DATA_DIR="/my_data/"
export DATA_DIR
source /home/eouser/flair_venv/bin/activate
which python
cd $HOME/exp_2026/LCC_Aer_SITS_FLAIR_HUB_MaxViT
python main.py --config_file=./configs/train_main/





