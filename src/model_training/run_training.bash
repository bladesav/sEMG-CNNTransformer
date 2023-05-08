#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=60G
#SBATCH --account=def-xilinliu
#SBATCH --gpus-per-node=1      # Number of GPU(s) per node
#SBATCH --cpus-per-task=6         # CPU cores/threads     
#SBATCH --time=2-0:00
#SBATCH --output=logs/%j.out

module load python
module load scipy-stack

cd ~/projects/sEMG-CNNTransformer
source newENV/bin/activate
python src/model_training/training_main.py 1 $SLURM_JOB_ID