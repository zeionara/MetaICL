#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --mem=32GB

#SBATCH --job-name=wdc-v3

## This places the standard output and standard error into the same file, in this case slurm_<job_id>.out 
#SBATCH --output=/scratch/jc11431/slurm_logs/slurm_%A.out
USERDIR=/home/jc11431 # This would be $HOME, except that I am sharing the same HOME dir across multiple users

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

SLICE="50"

# cd $SCRATCH/MetaICL/data/wdc-v3
# wget http://data.dws.informatik.uni-mannheim.de/webtables/2015-07/englishCorpus/compressed/$SLICE.tar.gz
# tar -xvf $SLICE.tar.gz

cd $USERDIR/git/MetaICL/wdc_setup
# python filter_slice_to_longlist.py --tarfile /scratch/jc11431/MetaICL/data/wdc-v3/$SLICE.tar
# python export_tasks_from_tables.py --tarfile /scratch/jc11431/MetaICL/data/wdc-v3/$SLICE.tar --tables_file /scratch/jc11431/MetaICL/data/wdc-v3/$SLICE/longlist.jsonl
python cluster_list_of_tables.py --input_dir /scratch/jc11431/MetaICL/data/wdc-v3/exports/ --num_clusters 50

echo "EXPORT COMPLETE. Exported to /scratch/jc11431/MetaICL/data/wdc-v3/$SLICE/longlist"