# Barlow Twin Pretraining of VMamba

## Requirements

We recommend setting up a conda environment and installing dependencies via pip. Use the following commands to set up your environment:
We recommend using the pytorch>=2.0, cuda>=11.8. But lower version of pytorch and CUDA are also supported. See the official VMamba repository for more details. 

***Create and activate a new conda environment***

```bash
conda create -n vmamba
conda activate vmamba
```

***Install Dependencies***

```bash
pip install -r requirements.txt
cd kernels/selective_scan && pip install .
```

## Training

Use the bash script train.sh to submit a training job. Replace /path/to/dataset to the your training data containing unlabelled images for pretraining. Prepare the dataset folder in the ImageFolder format as specified here, note that the class name does not matter as this is the pretraining phase [https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder]. To achieve a batch size of 64, 4 A100 GPUS were used.  

