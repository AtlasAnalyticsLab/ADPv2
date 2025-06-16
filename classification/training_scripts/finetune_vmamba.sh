''' This bash script runs a finetuning job for the VMamba model on a SLURM cluster. You may choose to run finetuning directly,
or after pretraining the model on an unlabelled dataset. To run pretraining, refer to pretrain_vmamba.sh.
Place pretrained weights in the specified checkpoint directory and update the path in the arguments below.
'''

#!/bin/bash
#SBATCH --gpus-per-node=4
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/YOURUSER/scratch/vmamba_finetune_%j.out
#SBATCH --account=ACC_NAME

# Show some cluster info (optional, but useful for debugging)
nvidia-smi
nvcc --version

# Load your virtualenv or conda environment
source ~/scratch/VmambaENV/bin/activate

# Print loaded modules
module list

FINETUNE_SCRIPT=/home/YOURUSER/scratch/VMamba/classification/vmamba_finetune.py
DATA_ROOT=/home/YOURUSER/scratch/histology_data
CFG_FILE=/home/YOURUSER/projects/def-msh-ab/YOURUSER/VMamba/classification/configs/vssm/vmambav0_base_224.yaml

IMG_DIR=/home/YOURUSER/scratch/all_colon_1360/
META_PATH=/home/YOURUSER/scratch/colon_metadata_complete.csv
TRAIN_CSV=/home/YOURUSER/scratch/train_test_80_20/train_annotations_16_tcga.csv
VAL_CSV=/home/YOURUSER/scratch/train_test_80_20/val_annotations_16_tcga.csv
TEST_CSV=/home/YOURUSER/scratch/train_test_80_20/test_annotations_16_tcga.csv
CKPT_DIR=/home/YOURUSER/scratch/vmamba_ckpt_finetune
CKPT_SAVE_DIR=/home/YOURUSER/scratch/vmamba_ckpt_save_finetune

python $FINETUNE_SCRIPT \
    $DATA_ROOT \
    --cfg $CFG_FILE \
    --img-dir $IMG_DIR \
    --meta-path $META_PATH \
    --train-csv $TRAIN_CSV \
    --val-csv $VAL_CSV \
    --test-csv $TEST_CSV \
    --checkpoint-dir $CKPT_DIR \
    --checkpoint-save-dir $CKPT_SAVE_DIR \
    --epochs 80 \
    --batch-size 128
