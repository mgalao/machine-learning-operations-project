# 🕵️‍♂️💳 Credit card fraud detection using MLOps

## 📋 Project Overview
This repository contains materials for a Machine Learning Operations project, that aims to study Fraud detection in the United States from 2019 to June 2020 and is focused on generating a pipeline that is capable of building models ready for production in a real case scenario, using tools like Kedro, MLFlow, Hopsworks, Evidently, NannyMl, Docker and more to build a full automatized machine learning pipeline.

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
