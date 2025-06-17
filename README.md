# Machine Learning Operations: Project

## 👥 Team Members
- Bruna Simões
- Daniel Caridade
- Leonardo Caterina
- Marco Galão

## 📋 Project Overview
This repository contains materials for a Machine Learning Operations project, that aims to study Fraud detection in the United States in October 2019.

## 🗂️ Repository Structure
```text
.
├── data/                               # Main data directory for all dataset stages and data-related scripts
│   ├── 01_raw/                         # Raw, unprocessed dataset
│   │   └── data_v1.csv                 # Original sample dataset of credit card transactions made in the USA
│   ├── 02_intermediate/                # Data after initial processing or cleaning steps
│   ├── 03_primary/                     # Primary dataset ready for modeling (cleaned, validated)
│   ├── 04_feature/                     # Feature-engineered datasets
│   ├── 05_model_input/                 # Final datasets used as input for model training
│   ├── 06_models/                      # Serialized models (e.g., .pkl, .joblib files)
│   ├── 07_model_output/                # Model predictions and output files
│   ├── 08_reporting/                   # Reports, evaluation results, visualizations
│   ├── data_loader.py                  # Script for loading datasets from the various folders
│── gx/                                 # Great Expectations data validation artifacts
│── mlruns/                             # MLflow experiment tracking directory
├── notebooks/                          # Notebooks used for experimentation and development
│   ├── EDA/                            # Exploratory Data Analysis notebooks
│   │   └── EDA code.ipynb              # Notebook performing EDA on the raw dataset
│──  optuna/                            # Notebooks or scripts for hyperparameter tuning with Optuna
├── src/                                # (Consider renaming or moving – 'src' should ideally be outside 'notebooks/')
├── streamlit/                          # Notebooks or files related to Streamlit dashboards/apps
├── README.md                           # Project documentation and overview
└── requirements.txt                    # Python package dependencies required to run the project
```
