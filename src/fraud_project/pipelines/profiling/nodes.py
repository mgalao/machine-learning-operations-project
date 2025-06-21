import logging
from typing import Optional

import pandas as pd
from ydata_profiling import ProfileReport
import great_expectations as gx

logger = logging.getLogger(__name__)

def get_validation_results(checkpoint_result):
    validation_result_key, validation_result_data = next(iter(checkpoint_result["run_results"].items()))
    validation_result_ = validation_result_data.get('validation_result', {})
    results = validation_result_["results"]
    meta = validation_result_["meta"]
    
    df_validation = pd.DataFrame(
        columns=[
            "Success","Expectation Type","Column","Column Pair","Max Value",
            "Min Value","Element Count","Unexpected Count","Unexpected Percent",
            "Value Set","Unexpected Value","Observed Value"
        ]
    )
    
    for result in results:
        success = result.get('success', '')
        expectation_type = result.get('expectation_config', {}).get('expectation_type', '')
        column = result.get('expectation_config', {}).get('kwargs', {}).get('column', '')
        column_A = result.get('expectation_config', {}).get('kwargs', {}).get('column_A', '')
        column_B = result.get('expectation_config', {}).get('kwargs', {}).get('column_B', '')
        value_set = result.get('expectation_config', {}).get('kwargs', {}).get('value_set', '')
        max_value = result.get('expectation_config', {}).get('kwargs', {}).get('max_value', '')
        min_value = result.get('expectation_config', {}).get('kwargs', {}).get('min_value', '')

        element_count = result.get('result', {}).get('element_count', '')
        unexpected_count = result.get('result', {}).get('unexpected_count', '')
        unexpected_percent = result.get('result', {}).get('unexpected_percent', '')
        observed_value = result.get('result', {}).get('observed_value', '')
        if isinstance(observed_value, list):
            unexpected_value = [item for item in observed_value if item not in value_set]
        else:
            unexpected_value = []

        df_validation = pd.concat([
            df_validation,
            pd.DataFrame.from_dict([{
                "Success": success,
                "Expectation Type": expectation_type,
                "Column": column,
                "Column Pair": (column_A, column_B),
                "Max Value": max_value,
                "Min Value": min_value,
                "Element Count": element_count,
                "Unexpected Count": unexpected_count,
                "Unexpected Percent": unexpected_percent,
                "Value Set": value_set,
                "Unexpected Value": unexpected_value,
                "Observed Value": observed_value
            }])
        ], ignore_index=True)
    
    return df_validation


def profile_and_validate_data(
    df: pd.DataFrame,
    context_root_dir: str = "../gx",
    datasource_name: str = "profiling_datasource",
    data_asset_name: str = "profiling_asset",
    suite_name: str = "profiling_suite",
    checkpoint_name: str = "profiling_checkpoint",
) -> pd.DataFrame:
    """
    Generate a ydata_profiling report, convert it to a Great Expectations suite, run validation,
    and return validation results as a DataFrame.
    """
    # Generate ProfileReport with minimal=True (can be changed as needed)
    profile = ProfileReport(df, title="Profiling Report", minimal=True)
    logger.info("ProfileReport generated")

    # Get GE context
    context = gx.get_context(context_root_dir=context_root_dir)
    
    # Add or get datasource
    try:
        datasource = context.sources.add_pandas(datasource_name)
        logger.info(f"Datasource '{datasource_name}' created.")
    except Exception:
        logger.info(f"Datasource '{datasource_name}' already exists.")
        datasource = context.datasources[datasource_name]

    # Add or get dataframe asset
    try:
        data_asset = datasource.add_dataframe_asset(name=data_asset_name, dataframe=df)
        logger.info(f"Data asset '{data_asset_name}' created.")
    except Exception:
        logger.info(f"Data asset '{data_asset_name}' already exists. Loading existing asset.")
        data_asset = datasource.get_asset(data_asset_name)

    # Build batch request
    batch_request = data_asset.build_batch_request(dataframe=df)

    # Convert profiling report to GE expectation suite
    new_suite = profile.to_expectation_suite(
        datasource_name=datasource_name,
        data_asset_name=data_asset_name,
        suite_name=suite_name,
        data_context=context,
    )
    logger.info(f"Expectation suite '{suite_name}' created from profiling report.")

    # Save expectation suite
    context.save_expectation_suite(expectation_suite=new_suite)

    # Setup and run checkpoint for validation
    checkpoint = gx.checkpoint.SimpleCheckpoint(
        name=checkpoint_name,
        data_context=context,
        validations=[
            {
                "batch_request": batch_request,
                "expectation_suite_name": suite_name,
            },
        ],
    )
    checkpoint_result = checkpoint.run()
    logger.info(f"Checkpoint '{checkpoint_name}' run completed.")

    # Get validation results as DataFrame
    df_validation = get_validation_results(checkpoint_result)
    
    logger.info("Validation results extracted.")
    return df_validation