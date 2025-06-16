''' This script is designed to run a pretraining job for the VMamba model using Barlow Twins on a SLURM cluster.
For effective pretraining results, prepare your unlabelled dataset in the specified directory and ensure the configuration file is correctly set up.
'''

#!/bin/bash
#SBATCH --gpus-per-node=4
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/home/YOURUSER/scratch/vmamba_barlowtwins_%j.out
#SBATCH --account=rrg-msh-ab

# Print GPU and CUDA information
nvidia-smi
nvcc --version

# Load your environment (adjust this line to your setup)
source ~/scratch/VmambaENV/bin/activate

# Print loaded modules for debug
module list

# Path to your refactored script
SCRIPT_PATH=/home/YOURUSER/scratch/VMamba/classification/vmamba_barlowtwins_pretrain.py

# Path to your data and config file
DATA_DIR=/home/YOURUSER/scratch/histology_data
CONFIG_FILE=/home/YOURUSER/projects/def-msh-ab/YOURUSER/VMamba/classification/configs/vssm/vmambav0_base_224.yaml

# Output directory for checkpoints
CHECKPOINT_DIR=/home/YOURUSER/scratch/vmamba_checkpoints_run1

# You can adjust batch size, epochs, etc. as needed!
python $SCRIPT_PATH \
  $DATA_DIR \
  --cfg $CONFIG_FILE \
  --checkpoint-dir $CHECKPOINT_DIR \
  --epochs 100 \
  --batch-size 64 \
  --workers 8 \
  --weight-decay 1e-6 \
  --learning-rate-weights 0.2 \
  --learning-rate-biases 0.0048 \
  --lambd 0.0051 \
  --print-freq 100

# Add any additional arguments as desired!
