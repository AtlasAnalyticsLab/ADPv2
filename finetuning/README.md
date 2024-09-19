# Finetuning of VMamba

## Requirements

Use same dependencies and environment as speicified in the README.md in the pretraining folder. 

## Training

Use the bash script train.sh to submit a training job. Replace /path/to/dataset to the your training data containing unlabelled images for pretraining. To achieve a batch size of 64, 4 A100 GPUS were used.  

