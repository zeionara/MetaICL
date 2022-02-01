#!/bin/bash
set -x

for setting_name in hr_to_lr class_to_class non_class_to_class qa_to_qa non_qa_to_qa non_nli_to_nli non_paraphrase_to_paraphrase
do
    # Multi-task 0-shot baselines
    sbatch train.sbatch $setting_name multitask-zero
    sbatch train.sbatch $setting_name channel-multitask-zero

    # MetaICL
    sbatch train.sbatch $setting_name metaicl
    sbatch train.sbatch $setting_name channel-metaicl
done