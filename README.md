# COF-ML

This repository contains the dataset and machine learning workflows developed for predicting the electrochemical properties of covalent organic frameworks (COFs), including open-circuit voltage (OCV) and gravimetric capacity.

## Repository Contents

### Dataset

- **`dice_wOCV_wcap.xlsx`**
  - DFT-derived dataset of OCV and theoretical capacities for the **Diverse Collection of Electroactive COFs (DICE)**, curated from the CORE COF database.

### Feature Generation

- **`feats.ipynb`**
  - Jupyter notebook for generating chemical and structural descriptors from the unrelaxed CIF structures of COFs.
  - Produces the feature matrix that serve as input to train the classifiers

### Machine Learning Models

- **`xgboost_ocv_classifier.py`**
  - Python script for training XGBoost models to:
    - classify COFs as **anodic** or **cathodic**, and
    - perform downstream OCV prediction tasks.

- **`xgboost_cap_classifier.py`**
  - Python script for training XGBoost models to:
    - classify COFs as **high-capacity** or **low-capacity**, and
    - perform downstream capacity prediction tasks.

## Usage

### Train the OCV classifier

```bash
python xgboost_ocv_classifier.py
```

### Train the capacity classifier

```bash
python xgboost_cap_classifier.py
```
## Citation

If you use this repository in your research, please cite:

**P. Bhauriyal, S. Das, and T. Heine**,  
*When are Two-Dimensional Covalent Organic Frameworks Good Electrodes for Lithium Ion Batteries? A Data-Driven Analysis*,  
**Advanced Science** (2026).  
DOI: [10.1002/advs.78599](https://doi.org/10.1002/advs.78599)

You can also use the **Cite this repository** option on GitHub to obtain the citation in BibTeX format.
