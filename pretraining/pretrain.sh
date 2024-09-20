#!/bin/bash

#SBATCH --job-name=BT
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node v100:8
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --time=8:00:00
#SBATCH --output=BT1/BT_randstain_output_%j.txt
#SBATCH --error=BT1/BT_randstain_error_%j.txt
#SBATCH --mem=250G

#SBATCH --account=acc_name
module load opencv
nvidia-smi 
source vmamba/bin/activate 
python pretrain.py /path/to/dataset
  --cfg /configs/vssm/vmambav0_base_224.yaml \
  --workers 8 \
  --epochs 100 \
  --batch-size 104 \
  --learning-rate-weights 0.2 \
  --learning-rate-biases 0.0048 \
  --weight-decay 1e-06 \
  --lambd 0.0048 \
  --projector 8192-8192-8192 \
  --print-freq 100 \
  --checkpoint-dir BT1/randstain_1
  --learning-rate 6e-05
deactivate
