#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=48:00:00
#SBATCH --mem=128GB

#SBATCH --job-name=metaicl-train

## This places the standard output and standard error into the same file, in this case slurm_<job_id>.out 
#SBATCH --output=/scratch/jc11431/slurm_logs/slurm_%A.out
USERDIR=/home/jc11431 # This would be $HOME, except that I am sharing the same HOME dir across multiple users

## First we ensure a clean environment by purging the current one
module purge

## Load Anaconda
module load anaconda3/2020.07
module load cuda/10.2.89
source ~/.bashrc
conda activate metaicl

## Just log environment stats for diagnostics
myquota
nvidia-smi
which python

cd $USERDIR/git/MetaICL

# hr_to_lr
# class_to_class
# non_class_to_class
# qa_to_qa
# non_qa_to_qa
# non_nli_to_nli
# non_paraphrase_to_paraphrase
task=$1
method=$2
echo "TASK: " $task
echo "METHOD: " $method


# metaicl
if [[ $method == "metaicl" ]] ; then
  echo "\n\n\nTensorizing..."
  time python train.py --task $task --k 16384 --test_k 16 --seed 100 --use_demonstrations --method direct --do_tensorize --n_gpu 1 --n_process 8

  echo "\n\n\nTraining..."
  time python train.py --task $task --k 16384 --test_k 16 --seed 100 --use_demonstrations --method direct --train_seed 1 --n_gpu 1 \
    --batch_size 1 --lr 1e-05 --fp16 --optimization 8bit-adam --out_dir checkpoints/metaicl/$task

  echo "Completed metaicl"
fi

# channel metaicl
if [[ $method == "channel-metaicl" ]] ; then
  echo "\n\n\nTensorizing..."
  time python train.py --task $task --k 16384 --test_k 16 --seed 100 --use_demonstrations --method channel --do_tensorize --n_gpu 1 --n_process 8

  echo "\n\n\nTraining..."
  time python train.py --task $task --k 16384 --test_k 16 --seed 100 --use_demonstrations --method channel --train_seed 1 --n_gpu 1 \
    --batch_size 1 --lr 1e-05 --fp16 --optimization 8bit-adam --out_dir checkpoints/channel-metaicl/$task

  echo "Completed channel metaicl"
fi

# multitask-zero
if [[ $method == "multitask-zero" ]] ; then
  echo "\n\n\nTensorizing..."
  time python train.py --task $task --k 16384 --seed 100 --method direct --do_tensorize --n_gpu 1 --n_process 8

  echo "\n\n\nTraining..."
  time python train.py --task $task --k 16384 --seed 100 --method direct --train_seed 1 --n_gpu 1 \
    --batch_size 1 --lr 1e-05 --fp16 --optimization 8bit-adam --out_dir checkpoints/multitask-zero/$task

  echo "Completed multitask-zero"
fi

# channel multitask-zero
if [[ $method == "channel-multitask-zero" ]] ; then
  echo "\n\n\nTensorizing..."
  time python train.py --task $task --k 16384 --seed 100 --method channel --do_tensorize --n_gpu 1 --n_process 8

  echo "\n\n\nTraining..."
  time python train.py --task $task --k 16384 --seed 100 --method channel --train_seed 1 --n_gpu 1 \
    --batch_size 1 --lr 1e-05 --fp16 --optimization 8bit-adam --out_dir checkpoints/channel-multitask-zero/$task

  echo "Completed channel multitask-zero"
fi
