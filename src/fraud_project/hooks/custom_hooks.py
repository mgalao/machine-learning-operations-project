from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.pipeline import Pipeline
import logging

logger = logging.getLogger(__name__)

class DriftHook:
    @hook_impl
    def after_pipeline_run(self, run_params, catalog: DataCatalog):
        """Triggered after pipeline run finishes."""
        if "drifted_features" in catalog.list():
            drifted_features = catalog.load("drifted_features")

            if drifted_features and len(drifted_features) > 0:
                logger.warning("Drift detected. Triggering retraining...")
                
                # Load retraining pipeline and run it
                from train_model.pipeline import create_pipeline as train_pipeline
                from kedro.runner import SequentialRunner

                train_model = train_pipeline()
                runner = SequentialRunner()
                runner.run(train_model, catalog=catalog)
