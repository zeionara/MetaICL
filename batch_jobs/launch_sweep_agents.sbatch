#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=48:00:00
#SBATCH --mem=64GB

#SBATCH --array=1-10
#SBATCH --job-name=sweep_job

## This places the standard output and standard error into the same file, in this case slurm_<job_id>_<arr_idx>.out 
#SBATCH --output=/scratch/jc11431/slurm_logs/slurm_%A_%a.out

## First we ensure a clean environment by purging the current one
module purge

## Load Anaconda
module load anaconda3/2020.07
source ~/.bashrc
conda activate alignment

## Just log environment stats for diagnostics
myquota
nvidia-smi
which python
wandb login

sweep_path=$1

## Run experiment
cd $HOME/git/MetaICL
echo SLURM_JOBID $SLURM_JOBID
echo SLURM_ARRAY_JOB_ID $SLURM_ARRAY_JOB_ID
echo SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID
wandb agent --count 1 $sweep_path