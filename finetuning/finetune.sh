#!/bin/bash

#SBATCH --job-name=Vmamba_finetune
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --time=1:30:00
#SBATCH --output=finetuning/finetune_outputs.txt
#SBATCH --error=finetuning/finetune_errors.txt
#SBATCH --mem=128G

#SBATCH --account=acc_name
nvidia-smi 
source vmamba/bin/activate 
python vmamba_finetune.py
  --cfg path/to/config \
  --checkpoint-dir path/to/checkpoint.pth \
  --output-dir finetuning/output_dir
deactivate
