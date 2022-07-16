#!/bin/bash
set -x

for setting_name in hr_to_lr class_to_class non_class_to_class qa_to_qa non_qa_to_qa non_nli_to_nli non_paraphrase_to_paraphrase
do
    # # raw LM zero-shot baselines (0-shot, PMI 0-shot, Channel 0-shot)
    # sbatch evaluate.sbatch $setting_name zero 100 64
    # sbatch evaluate.sbatch $setting_name pmi-zero 100 64
    # sbatch evaluate.sbatch $setting_name channel-zero 100 64

    # # raw LM in-context baselines (in-context, PMI in-context, Channel in-context)
    # sbatch evaluate.sbatch $setting_name ic 100,13,21,42,87 16
    # sbatch evaluate.sbatch $setting_name pmi-ic 100,13,21,42,87 16
    # sbatch evaluate.sbatch $setting_name channel-ic 100,13,21,42,87 16

    # # Multi-task 0-shot baselines
    # sbatch evaluate.sbatch $setting_name multitask-zero 100 64
    # sbatch evaluate.sbatch $setting_name channel-multitask-zero 100 64

    # # MetaICL
    # sbatch evaluate.sbatch $setting_name metaicl 100,13,21,42,87 16
    # sbatch evaluate.sbatch $setting_name channel-metaicl 100,13,21,42,87 16

    # # Trained Multi-task 0-shot baselines
    # sbatch evaluate.sbatch $setting_name multitask-zero 100 64 model-30000.pt
    # sbatch evaluate.sbatch $setting_name channel-multitask-zero 100 64 model-30000.pt

    # # Trained MetaICL
    # sbatch evaluate.sbatch $setting_name metaicl 100,13,21,42,87 16 model-30000.pt
    # sbatch evaluate.sbatch $setting_name channel-metaicl 100,13,21,42,87 16 model-30000.pt

    # Custom Multi-task 0-shot baselines
    sbatch evaluate.sbatch $setting_name multitask-zero 100 64 /home/jc11431/git/MetaICL/checkpoints/best-metaicl/20185967_2-best_dev_score.pt
    sbatch evaluate.sbatch $setting_name channel-multitask-zero 100 64 /home/jc11431/git/MetaICL/checkpoints/best-metaicl/20185967_2-best_dev_score.pt

    # Custom MetaICL
    sbatch evaluate.sbatch $setting_name metaicl 100,13,21,42,87 16 /home/jc11431/git/MetaICL/checkpoints/best-metaicl/20185967_2-best_dev_score.pt
    sbatch evaluate.sbatch $setting_name channel-metaicl 100,13,21,42,87 16 /home/jc11431/git/MetaICL/checkpoints/best-metaicl/20185967_2-best_dev_score.pt
done