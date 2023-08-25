#!/bin/sh
#
# Simple "Hello World" submit script for Slurm.
#
# Replace ACCOUNT with your account name before submitting.
#
#SBATCH --account=dsi               
#SBATCH --job-name=backtranslate    # The job name
#SBATCH -N 1                        # The number of nodes to request
#SBATCH -c 1                        # The number of cpu cores to use (up to 32 cores per server)
#SBATCH --mem-per-cpu=32G           # The memory the job will use per cpu core
#SBATCH --gres=gpu:1
#SBATCH --time=0-0:15               # The time the job will take to run in D-HH:MM

source ~/.bashrc

conda activate cysyphus-ml

python back_translate.py russian
 
# End of script