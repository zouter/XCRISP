# X-CRISP: Domain-Adaptable and Interpretable CRISPR Repair Outcome Prediction
[BioArXiv paper](https://doi.org/10.1101/2025.02.06.636858)


==============================

## Abstract

**Motivation:** Controlling the outcomes of CRISPR editing is crucial for the success of gene therapy. Since donor template-based editing is often inefficient, alternative strategies have emerged that leverage mutagenic end-joining repair instead. Existing machine learning models can accurately predict end-joining repair outcomes, however: generalisability beyond the specific cell line used for training remains a challenge, and interpretability is typically limited by suboptimal feature representation and model architecture.

**Results:** We propose X-CRISP, a flexible and interpretable neural network for predicting repair outcome frequencies based on a minimal set of outcome and sequence features, including microhomologies (MH). Outperforming prior models on detailed and aggregate outcome predictions, X-CRISP prioritised MH location over MH sequence properties such as GC content for deletion outcomes. Through transfer learning, we adapted X-CRISP pre-trained on wild-type mESC data to target human cell lines K562, HAP1, U2OS, and mESC lines with altered DNA repair function. Adapted X-CRISP models improved over direct training on target data from as few as 50 samples, suggesting that this strategy could be leveraged to build models for new domains using a fraction of the data required to train models from scratch.

This repository contains all the code used to process the data and generate the results.

Directory Structure
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── batch              <- Slurm scripts 
    │
    ├── containers         <- Apptainer container definition files.
    |
    ├── docs               <- Unused
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── slurm              <- Slurm batch files and shell scripts to execute multiple batch operations at once
    │
    ├── scripts            <- Apptainer scripts for building, deployment, etc. Lima scripts for VM startup and shutdown for buidling apptainer.
    │
    ├── references         <- Unused
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Ad-hoc analysis
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to configure training and testing experiments  
        ├── data           <- Scripts to download or generate data    
        ├── features       <- Scripts to turn raw data into features for modeling    
        ├── models         <- Scripts to train models and then use trained models to make predictions
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
        └── preprocessing  <- Scripts to preprocess data


--------
