#!/bin/bash 
#SBATCH --job-name=LCC_Aer_SITS_FLAIR_HUB_MaxViT
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100m40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=48:00:00
#SBATCH --mail-user=kanyamahanga@ipi.uni-hannover.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output logs/LCC_Aer_SITS_FLAIR_HUB_MaxViT_%j.out
#SBATCH --error logs/LCC_Aer_SITS_FLAIR_HUB_MaxViT_%j.err
source reqrts/load_modules.sh
export CONDA_ENVS_PATH=$SOFTWARE/.conda/envs
export DATA_DIR=$BIGWORK
export TRANSFORMERS_OFFLINE=1
conda activate /software/NHGN20600/nhgnkany/flair_venv
which python
cd $HOME/LCC_Aer_SITS_FLAIR_HUB_MaxViT
srun python main.py --config_file=./configs/train_main/







