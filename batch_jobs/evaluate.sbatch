#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=100:00:00
#SBATCH --mem=64GB

#SBATCH --job-name=metaicl-reproduce-results

## This places the standard output and standard error into the same file, in this case slurm_<job_id>_<arr_idx>.out 
#SBATCH --output=/scratch/jc11431/slurm_logs/slurm_%A_%a.out

## First we ensure a clean environment by purging the current one
module purge

## Load Anaconda
module load anaconda3/2020.07
source ~/.bashrc
conda activate metaicl

## Just log environment stats for diagnostics
myquota
nvidia-smi
which python

cd $HOME/git/MetaICL

task=$1
method=$2
seed=$3
bs=$4
model=$5
echo "Running " $task $method $seed $bs $model
time bash reproduce.sh $task $method $seed $bs $model

echo "COMPLETED: $task $method $seed $bs $model"