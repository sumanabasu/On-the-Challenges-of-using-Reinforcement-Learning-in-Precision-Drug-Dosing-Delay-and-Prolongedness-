#!/bin/bash

#SBATCH --array=0-14
#SBATCH -c 1
#SBATCH --mem=16g
#SBATCH --gres=gpu:1
#SBATCH --time=1-23:00:00
#SBATCH --job-name=Diabetes_ADRQN
#SBATCH --error=/home/mila/b/basus/AAAIcode/Experiments/errors/error-%A_%a.err
#SBATCH -o /home/mila/b/basus/AAAIcode/Experiments/logs/slurm-%A_%a.out

# 1. Load the required modules
# mila-cluster specific module:
module --quiet load anaconda/3

# 2. Load your environment
conda activate diabetes_pomdp

# 3. load pytorch
module --quiet load pytorch/1.8.1

cd /home/mila/b/basus/AAAIcode/paepomdp/diabetes/mains || exit
date;hostname;pwd
python adrqn_diabetes_main.py --array_id=$SLURM_ARRAY_TASK_ID --patient_name=child#009
date
nvidia-smi