#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --mem=32GB

#SBATCH --job-name=wdc-v5

## This places the standard output and standard error into the same file, in this case slurm_<job_id>.out 
#SBATCH --output=/scratch/jc11431/slurm_logs/slurm_%A.out
USERDIR=/home/jc11431 # This would be $HOME, except that I am sharing the same HOME dir across multiple users

## First we ensure a clean environment by purging the current one
module purge

## Load Anaconda
module load anaconda3/2020.07
source ~/.bashrc
conda activate $USERDIR/.conda/envs/metaicl

## Just log environment stats for diagnostics
myquota
nvidia-smi
which python

WDC_VERSION="v5"

cd /home/jc11431/git/MetaICL/data
zip -r wdc-$WDC_VERSION.zip wdc-$WDC_VERSION