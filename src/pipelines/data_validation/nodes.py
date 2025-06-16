import pandas as pd

# Function to split the data to use 2019 transactions as references and the rest for validation
def split_data_by_timestamp(df: pd.DataFrame, timestamp_col: str = "Timestamp"):
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    reference = df[df[timestamp_col].dt.year == 2019].copy()
    validation = df[df[timestamp_col].dt.year > 2019].copy()
    return reference, validation

from typing import Optional
import pandas as pd
from ydata_profiling import ProfileReport
from your_module import ExpectationsReportV3  # The class you gave, ideally in a utils.py

# Create a function to generate the expectations
def generate_expectations_and_validate(
    reference_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    gx_root_dir: str,
    datasource_name: str = "pandas_datasource",
    data_asset_name: str = "my_data_asset",
    suite_name: Optional[str] = None,
):
    import great_expectations as ge
    from great_expectations.data_context import DataContext
    
    # 1. Initialize GX context
    data_context = DataContext(gx_root_dir)
    
    # 2. Generate expectations suite from reference dataset using your class
    report = ExpectationsReportV3()
    report.df = reference_df
    # You may want to configure the report's config if needed, e.g., report.config = ...
    
    expectation_suite = report.to_expectation_suite(
        datasource_name=datasource_name,
        data_asset_name=data_asset_name,
        suite_name=suite_name,
        data_context=data_context,
        save_suite=True,
        run_validation=False,  # skip validation on reference, just create suite
        build_data_docs=True,
    )
    
    # 3. Validate validation_df against the generated suite
    
    # Build batch request with runtime batch for validation data
    batch_request = {
        "datasource_name": datasource_name,
        "data_connector_name": "default_runtime_data_connector_name",
        "data_asset_name": data_asset_name,
        "runtime_parameters": {"batch_data": validation_df},
        "batch_identifiers": {"default_identifier_name": "validation_batch"},
    }
    
    validator = data_context.get_validator(
        batch_request=batch_request,
        expectation_suite=expectation_suite
    )
    
    validation_result = validator.validate()
    
    # Save validation result JSON for reporting
    validation_result.save_result()
    
    # Optional: build data docs after validation
    data_context.build_data_docs()
    
    if not validation_result["success"]:
        raise AssertionError("Validation failed on validation dataset!")
    
    return expectation_suite, validation_result

