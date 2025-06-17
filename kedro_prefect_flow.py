from prefect import flow, task, get_run_logger
from src.bank_full_project.run_kedro_pipeline import run_pipeline
import sys
import os

# Fix for emoji/log encoding issues on Windows
sys.stdout.reconfigure(encoding="utf-8")

# Ensure src path is accessible
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

@task
def run_kedro_task(pipeline_name: str):
    logger = get_run_logger()
    logger.info(f"Running Kedro pipeline: `{pipeline_name}`")

    try:
        run_pipeline(pipeline_name)
        logger.info(f"Kedro pipeline `{pipeline_name}` finished successfully.")
    except Exception as e:
        logger.error(f"Pipeline `{pipeline_name}` failed: {str(e)}")
        raise




@flow(name="Data Unit Test Flow")
def flow_data_unit_tests():
    run_kedro_task("data_unit_tests")

@flow(name="Full Training Processing Flow", description="Runs full preprocessing pipeline")
def flow_full_processing():
    run_kedro_task("preprocess_train")

@flow(name="Training Flow")
def flow_train():
    run_kedro_task("model_train")

@flow(name="Orchestration Flow")
def full_pipeline():
    flow_full_processing() 
    flow_train() 
