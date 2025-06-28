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
├── .viz/                    # Auto-generated folder used by Kedro Viz to store visualization state (e.g., node metadata).
├── app/                     # Contains code for deployment and API serving.
├── conf/                    # Configuration files for pipelines, data catalog, parameters, credentials, and environment settings.
├── data/                    # Directory for raw, intermediate, and processed datasets. Structured as per Kedro’s data layers: 01_raw, 02_intermediate, 03_primary, etc.
├── docs/                    # Documentation for the project.
│── gx/                      # Folder used by Great Expectations for data quality checks and expectations.
│── mlruns/                  # MLflow experiment tracking directory
├── notebooks/               # Notebooks used for experimentation and development
├── src/                     # (Consider renaming or moving – 'src' should ideally be outside 'notebooks/')
├── tests/                   # Notebooks or files related to Streamlit dashboards/apps
├── .coverage                # Auto-generated file used by coverage tools to track code test coverage.
├── .dockerignore            # Specifies files and directories that should be excluded from the Docker build context (like .git, __pycache__, etc.).
├── .gitignore               # Defines intentionally untracked files to ignore in Git, helping keep the repository clean.
├── Dockerfile               # Contains instructions to build a Docker image for the project.
├── README.md                # Main markdown file describing the project — purpose, setup instructions, usage, etc.
├── info.log                 # Likely a generated log file capturing runtime info or debugging messages.
├── kedro_viz.cmd            # A Windows batch script to run Kedro-Viz, the interactive pipeline visualization tool.
├── pyproject.toml           # A configuration file that defines your Python project metadata, dependencies, and build system.
├── requirements.txt         # Lists the core Python dependencies needed to run your project.
├── requirements-serving.txt # Specifies extra dependencies required for model serving (e.g., FastAPI, gunicorn, etc.).
└── uv.lock                  # Lock file created by uv, a fast Python package manager. Ensures reproducible installs.
```

## 🚀 Getting Started

Follow these steps to clone and use this repository:

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```
2. Install uv
   ```bash
   pip install uv
   ```
3. Install the requirements to run the project:
  - With pip only
   ```bash
   pip install -r requirements.txt
   ```
  - Or with uv (recommended):
   ```bash
   uv pip install -r requirements.txt
   ```
4. Explore the project and make changes as you see fit
   - Navigate through the pipelines.
   - Modify components as needed to suit your use case.
   - Run the project using Kedro as orchestration tool
