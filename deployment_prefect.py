from prefect.deployments import Deployment
from prefect.client.schemas.schedules import CronSchedule
from kedro_prefect_flow import flow_data_unit_tests, full_pipeline, flow_full_processing, flow_train  # Adjust path as needed
#from kedro_prefect_flow import orchestrator
from prefect.infrastructure import Process

# Deployment 1: Data Unit Tests
Deployment.build_from_flow(
    flow=flow_data_unit_tests,
    name="unit-test-nightly",
    work_queue_name="default",
    schedule=CronSchedule(cron="0 22 * * *", timezone="Europe/Lisbon"),  # Every day at 10 PM Lisbon
).apply()

# Deployment 2: Full Data Processing
Deployment.build_from_flow(
    flow=flow_full_processing,
    name="full-processing-morning",
    work_queue_name="default",
    schedule=CronSchedule(cron="0 7 * * *", timezone="Europe/Lisbon"),
    tags=["training-preprocess"]  # Every day at 7 AM Lisbon
).apply()

# Deployment 3: Model Training
Deployment.build_from_flow(
    flow=flow_train,
    name="model-train-weekly",
    work_queue_name="default",
    schedule=CronSchedule(cron="0 6 * * 1", timezone="Europe/Lisbon"),  # Every Monday at 6 AM Lisbon
).apply()

Deployment.build_from_flow(
    flow=full_pipeline,
    name="full_pipeline",
    work_queue_name="default",        
    infrastructure=Process()          
).apply()