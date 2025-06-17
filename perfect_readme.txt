
Prefect Pipeline Orchestration - README

This guide walks you through setting up and running Prefect flows with a dashboard, agent, and scheduled deployments for a Kedro project.

Prerequisites

- Prefect installed
  Install Prefect:
  pip install prefect

- Project structure follows:

  your-project/
  ├── src/
  │   └── your_project/
  │       └── run_kedro_pipeline.py
  ├── flows/
  │   └── kedro_flows.py
  ├── create_deployments.py
  └── README.txt

 Step-by-Step Setup

1. Start Prefect UI

Launch the Prefect dashboard locally:
  prefect server start

The UI will be available at: http://127.0.0.1:4200

Leave this running in one terminal.

2. Start Prefect Agent

In a new terminal, start an agent to pick up flow runs:
  prefect agent start --pool default-agent-pool

Make sure your flows use the same pool (default-agent-pool).

3. Define Your Kedro Flows

Example: flows/kedro_flows.py

  from prefect import flow, task, get_run_logger
  import os, sys
  sys.stdout.reconfigure(encoding="utf-8")
  sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
  from bank_full_project.run_kedro_pipeline import run_pipeline

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

  @flow(name="Full Training Processing Flow")
  def flow_full_processing():
      run_kedro_task("preprocess_train_pipeline")

  @flow(name="Training Flow")
  def flow_reporting():
      run_kedro_task("model_train")

4. Create Scheduled Deployments

Create create_deployments.py:

  from prefect.deployments import Deployment
  from prefect.client.schemas.schedules import CronSchedule
  from flows.kedro_flows import flow_data_unit_tests, flow_full_processing, flow_reporting

  Deployment.build_from_flow(
      flow=flow_data_unit_tests,
      name="unit-test-nightly",
      work_queue_name="default",
      schedule=CronSchedule(cron="0 22 * * *", timezone="Europe/Lisbon"),
  ).apply()

  Deployment.build_from_flow(
      flow=flow_full_processing,
      name="full-processing-morning",
      work_queue_name="default",
      schedule=CronSchedule(cron="0 7 * * *", timezone="Europe/Lisbon"),
  ).apply()

  Deployment.build_from_flow(
      flow=flow_reporting,
      name="model-train-weekly",
      work_queue_name="default",
      schedule=CronSchedule(cron="0 6 * * 1", timezone="Europe/Lisbon"),
  ).apply()

Run the script:
  python create_deployments.py

You’ll now see the scheduled deployments in the UI.

 Manual Run

To manually trigger a deployment from the terminal:
  prefect deployment run "unit-test-nightly"

Or from the UI: Click the  button beside the deployment name.

 Health Check

- Ensure Prefect server is running
- Ensure agent is connected to the same work pool
- Use the UI to monitor and debug
- Logs will appear in the agent terminal and UI

 Troubleshooting
(Add more examples of troubleshooting)
- ModuleNotFoundError: Ensure src/ is in sys.path before imports
- Deployment not scheduled? Check cron syntax and timezone
- Flow not running? Confirm the agent is connected to the correct work queue

 Clean Up

To stop the system:
# Stop the server (Ctrl+C in terminal)
# Stop the agent (Ctrl+C in terminal)

To delete flows:
  prefect deployment delete <deployment_name>
