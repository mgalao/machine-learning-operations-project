# """Project pipelines."""
# from __future__ import annotations


# def register_pipelines() -> dict[str, Pipeline]:
#     """Register the project's pipelines.

#     Returns:
#         A mapping from pipeline names to ``Pipeline`` objects.
#     """
#     pipelines = find_pipelines()
#     pipelines["__default__"] = sum(pipelines.values())
#     return pipelines




"""Project pipelines."""
from typing import Dict
from kedro.pipeline import Pipeline

from fraud_project.pipelines import (
    ingestion,
    data_quality,
    split_data,
    split_train_pipeline,
    feature_selection,
    model_selection,
    model_train,
    model_predict,
    data_drift,
    batch_sample_drift
)
from fraud_project.pipelines.preprocessing import (
    preprocessing_train, 
    preprocessing_batch,
    load_preprocessed_fs
)

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    # Ingestion & Validation
    pipeline_ingest = ingestion.create_pipeline()
    pipeline_validate = data_quality.create_pipeline()

    # Data Preparation
    pipeline_split_raw = split_data.create_pipeline()
    pipeline_preprocess_train = preprocessing_train.create_pipeline()
    pipeline_preprocess_batch = preprocessing_batch.create_pipeline()
    pipeline_load_preprocessed_fs = load_preprocessed_fs.create_pipeline()

    # Training & Evaluation
    pipeline_split_train = split_train_pipeline.create_pipeline()
    pipeline_feature_selection = feature_selection.create_pipeline()
    pipeline_model_selection = model_selection.create_pipeline()
    pipeline_model_train = model_train.create_pipeline()

    # Inference
    pipeline_predict = model_predict.create_pipeline()

    # Monitoring
    pipeline_drift_detection = data_drift.create_pipeline()
    pipeline_batch_sample_drift = batch_sample_drift.create_pipeline()
 
    return {
        # Individual pipelines
        "ingest": pipeline_ingest,
        "validate": pipeline_validate,
        "split_raw_data": pipeline_split_raw,
        "preprocess_train": pipeline_preprocess_train,
        "preprocess_batch": pipeline_preprocess_batch,
        "load_preprocessed_fs": pipeline_load_preprocessed_fs,
        "split_train_data": pipeline_split_train,
        "feature_selection": pipeline_feature_selection,
        "model_selection": pipeline_model_selection,
        "train_model": pipeline_model_train,
        "predict": pipeline_predict,
        "monitor_drift": pipeline_drift_detection,
        "batch_sample_drift": pipeline_batch_sample_drift,

        # Combined pipelines
        "ingest_to_validate": (
            pipeline_ingest
            + pipeline_split_raw
            + pipeline_validate
        ),
        "preproc_to_model_selection": (
            pipeline_preprocess_train
            + pipeline_preprocess_batch
            + pipeline_split_train
            + pipeline_feature_selection
            + pipeline_model_selection
        ),
        "model_train_predict": (
            pipeline_model_train
            + pipeline_predict
        ),

        # Full production process
        "__default__": (
            pipeline_ingest
            + pipeline_split_raw
            + pipeline_validate
            + pipeline_preprocess_train
            + pipeline_preprocess_batch
            + pipeline_split_train
            + pipeline_feature_selection
            + pipeline_model_selection
            + pipeline_model_train
            + pipeline_predict
            + pipeline_drift_detection
        )
    }