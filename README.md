# Vision Mamba for ADPv2: Hierarchical Histological Tissue Type Classification

This project provides an implementation of the **Vision Mamba** model for training and evaluation on the **Atlas of Digital Pathology v2 (ADPv2)** dataset, as described in the paper:

> **ADPv2: A Hierarchical Histological Tissue Type-Annotated Dataset for potential Biomarker Discovery of Colorectal Disease** ([paper link])

The ADPv2 dataset comprises **20,000 image patches** from healthy colon tissue, each annotated according to a highly detailed **hierarchical tissue taxonomy** spanning **32 distinct tissue types** across three levels of tissue specificity. For a comprehensive description of the dataset, refer to Section 3 ("Dataset") of the paper.

---

## Structure

- **Pretraining Script:** Self-supervised pretraining of Vision Mamba using Barlow Twins on unlabelled histology patches.  
  _This pretraining phase is optional but recommended to enhance downstream classification performance._

- **Finetuning Script:** Supervised training on labelled ADPv2 tissue types for multi-label classification.  
  _This step uses the labelled patches as shown in the taxonomy below._

- **SLURM Scripts:** Sample job scripts for distributed GPU training on clusters like Compute Canada.

- **Taxonomy Reference:** See the table below for the hierarchical taxonomy of tissue types used in this project.  
  _Green labels indicate the tissue types included in the current multi-label model; red labels are defined in the ontology but do not appear in the present dataset._

---

## Hierarchical Taxonomy of Histological Tissue Types

**Table 1:** Hierarchical taxonomy of histological tissue types in ADPv2.

- **Red** = tissue types without any presence in the current dataset.
- **Green** = tissue types included in multi-label representation model training.

![ADPv2 Tissue Taxonomy Table](./Screenshot%202025-06-10%20at%204.22.52%E2%80%AFPM.png)

_For more information on each tissue class, refer to the paper and the image above._

---

## 🚀 Getting Started

### 1. Clone & Environment

```bash
git clone https://github.com/AtlasAnalyticsLab/ADPv2.git
cd ADPv2

# Create and activate environment
python -m venv VmambaEnv
source VmambaEnv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📂 Folder Structure

Download annotated dataset at the folowing links:
Part 1: https://zenodo.org/records/15307021
Part 2: https://zenodo.org/records/15312384
Part 3: https://zenodo.org/records/15312792
Simply place the downloaded dataset images under the same training folder. 
Likewise place corresponding ground truth annotation csv files into a single unified csv file. 
For specific training setup, refer to the bash scripts found in classifications/training_scripts/. configure the filepaths for the training images and ground truth files to your specific paths. 

📄 Citation
If you use this code, please cite:

[citation]

🤝 Contributing
Fork the repo

Create a feature branch (git checkout -b feat/awesome)

Commit your changes (git commit -m "Add awesome feature")

Open a Pull Request