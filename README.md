# 🕵️‍♂️💳 Credit card fraud detection using MLOps

## 📋 Project Overview
This repository contains the complete implementation of a Machine Learning Operations (MLOps) project focused on credit card fraud detection in the United States between 2019 and June 2020. The main objective is to design and develop an automated, production-ready ML pipeline capable of detecting fraudulent transactions in real-world conditions.

Built with modern MLOps practices, the pipeline leverages tools such as Kedro for orchestration, MLflow for experiment tracking and model versioning, Hopsworks for feature management, Evidently and NannyML for drift detection and data quality monitoring, and Docker for containerized deployment. The project brings together model training, evaluation, monitoring, and deployment into a seamless, reproducible workflow.

This end-to-end pipeline is designed not only to train and serve models, but also to continuously monitor data drift, retrain models when necessary, and support integration with real-time environments.

## 👥 Team Members
- Bruna Simões
- Daniel Caridade
- Leonardo Caterina
- Marco Galão

## 🗂️ Repository Structure
```text
.
├── .viz/         # Auto-generated folder used by Kedro Viz to store visualization state (e.g., node metadata).
├── app/          # Contains code for deployment and API serving.
├── conf/         # Configuration files for pipelines, data catalog, parameters, credentials, and environment settings.
├── data/         # Directory for raw, intermediate, and processed datasets. Structured as per Kedro’s data layers: 01_raw, 02_intermediate, 03_primary, etc.
├── docs/         # Documentation for the project.
│── gx/           # Folder used by Great Expectations for data quality checks and expectations.
│── mlruns/       # MLflow experiment tracking directory
├── notebooks/    # Notebooks used for experimentation and development
├── src/          # (Consider renaming or moving – 'src' should ideally be outside 'notebooks/')
├── tests/        # Notebooks or files related to Streamlit dashboards/apps
├── README.md                           # Project documentation and overview
└── requirements.txt                    # Python package dependencies required to run the project
```

## How to clone and use this repository

1. 
