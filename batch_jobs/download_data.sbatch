#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=64GB

#SBATCH --job-name=metaicl-download-data

## This places the standard output and standard error into the same file, in this case slurm_<job_id>.out 
#SBATCH --output=/scratch/jc11431/slurm_logs/slurm_%A.out

## First we ensure a clean environment by purging the current one
module purge

## Load Anaconda
module load anaconda3/2020.07
source ~/.bashrc
conda activate crossfit

## Just log environment stats for diagnostics
myquota
nvidia-smi
which python

cd $HOME/git/MetaICL

cd preprocess

# preprocess from unifiedqa
echo "STARTING UNIFIED_QA\n\n"
python unifiedqa.py --do_train --do_test # skip `--do_train` if you won't run training yourself

echo "STARTING TEST\n\n"
# preprocess from crossfit
python _build_gym.py --build --n_proc=40 --do_test
echo "STARTING TRAIN\n\n"
python _build_gym.py --build --n_proc=40 --do_train # skip if you won't run training yourself