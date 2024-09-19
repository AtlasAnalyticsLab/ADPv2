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

