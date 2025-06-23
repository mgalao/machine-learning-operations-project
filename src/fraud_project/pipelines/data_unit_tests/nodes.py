"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from great_expectations.core import ExpectationSuite, ExpectationConfiguration
import great_expectations as gx

from pathlib import Path

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings


logger = logging.getLogger(__name__)


def get_validation_results(checkpoint_result):
    # validation_result is a dictionary containing one key-value pair
    validation_result_key, validation_result_data = next(iter(checkpoint_result["run_results"].items()))

    # Accessing the 'actions_results' from the validation_result_data
    validation_result_ = validation_result_data.get('validation_result', {})

    # Accessing the 'results' from the validation_result_data
    results = validation_result_["results"]
    meta = validation_result_["meta"]
    use_case = meta.get('expectation_suite_name')
    
    
    df_validation = pd.DataFrame({},columns=["Success","Expectation Type","Column","Column Pair","Max Value",\
                                       "Min Value","Element Count","Unexpected Count","Unexpected Percent","Value Set","Unexpected Value","Observed Value"])
    
    
    for result in results:
        # Process each result dictionary as needed
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
        if type(observed_value) is list:
            #sometimes observed_vaue is not iterable
            unexpected_value = [item for item in observed_value if item not in value_set]
        else:
            unexpected_value=[]
        
        df_validation = pd.concat([df_validation, pd.DataFrame.from_dict( [{"Success" :success,"Expectation Type" :expectation_type,"Column" : column,"Column Pair" : (column_A,column_B),"Max Value" :max_value,\
                                           "Min Value" :min_value,"Element Count" :element_count,"Unexpected Count" :unexpected_count,"Unexpected Percent":unexpected_percent,\
                                                  "Value Set" : value_set,"Unexpected Value" :unexpected_value ,"Observed Value" :observed_value}])], ignore_index=True)
        
    return df_validation

def test_data(df: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    context = gx.get_context(context_root_dir="../gx")
    datasource_name = "fraud_datasource"
    target_col = parameters["target_column"]

    # Set up the data source
    try:
        datasource = context.sources.add_pandas(datasource_name)
        logger.info("Datasource created.")
    except Exception:
        logger.info("Datasource already exists.")
        datasource = context.datasources[datasource_name]

    suite = context.add_or_update_expectation_suite(expectation_suite_name="fraud")

    # Column existence
    expected_columns = [
        "cc_num", "merchant", "category", "first", "last", "gender", "street", "city", "state", "zip", "job",
        "merch_zipcode", "age", "amt", "lat", "long", "city_pop", "merch_lat", "merch_long", "datetime",
        "trans_num", target_col
    ]
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_table_columns_to_match_set",
            kwargs={"column_set": expected_columns, "exact_match": True},
        )
    )

    # ID Column Expectations
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_of_type",
            kwargs={"column": "trans_num", "type_": "object"},
        )
    )
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_unique",
            kwargs={"column": "trans_num"}, # cc_num can be non-unique, representing multiple transactions
        )
    )

    # Numerical Column Expectations
    numerical_expectations = [
        ("age", 16, 120),
        ("amt", 0, 1_000_000),
        ("lat", -90, 90),
        ("long", -180, 180),
        ("city_pop", 0, 100_000_000),
        ("merch_lat", -90, 90),
        ("merch_long", -180, 180),
    ]
    for col, min_v, max_v in numerical_expectations:
        int_columns = ['age', 'city_pop']
        expected_type = "int64" if col in int_columns else "float64"
        suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_of_type",
                kwargs={"column": col, "type_": expected_type},
            )
        )
        suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": col, "min_value": min_v, "max_value": max_v},
            )
        )
        # Define null tolerance for each column
        if col in {"amt", "age"}:
            # No nulls allowed
            null_kwargs = {"column": col}
        else:
            # Allow up to 5% nulls for other numerical columns
            null_kwargs = {"column": col, "mostly": 0.95}
            
        suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs=null_kwargs,
            )
        )

    # Categorical Expectations
    string_columns = [
        "cc_num", "merchant", "category", "first", "last", "gender",
        "street", "city", "state", "zip", "job", "merch_zipcode"
    ]
    for col in string_columns:
        suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_of_type",
                kwargs={"column": col, "type_": "object"},
            )
        )
        suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": col, "mostly": 0.95},
            )
        )

    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={"column": "gender", "value_set": ["M", "F", "U"]},
        )
    )

    expected_states = [
        'MO', 'CA', 'OR', 'PA', 'VA', 'NE', 'IA', 'IL', 'WV', 'OK', 'GA',
        'MN', 'NM', 'NY', 'NC', 'LA', 'ME', 'KS', 'NV', 'NJ', 'AL', 'TX',
        'SD', 'MA', 'MI', 'SC', 'MT', 'FL', 'MS', 'IN', 'UT', 'CT', 'KY',
        'OH', 'MD', 'ID', 'ND', 'TN', 'WI', 'VT', 'WY', 'CO', 'NH', 'AR',
        'WA', 'AZ', 'HI', 'DC', 'RI', 'AK', 'DE'
    ]
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={"column": "state", "value_set": expected_states},
        )
    )

    # Regex Expectations
    # DateTime: Expecting format 'YYYY-MM-DD HH:MM:SS'
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_of_type",
            kwargs={"column": "datetime", "type_": "datetime64[ns]"},
        )
    )
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_match_strftime_format",
            kwargs={"column": "datetime", "strftime_format": "%Y-%m-%d %H:%M:%S"},
        )
    )
    # Zip code: US ZIP codes typically 5 digits or 5+4 digits
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_match_regex",
            kwargs={
                "column": "zip",
                "regex": r"^\d{5}(-\d{4})?$"
            },
        )
    )
    # Credit Card Number: Simplified numeric check of 13 to 19 digits
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_match_regex",
            kwargs={
                "column": "cc_num",
                "regex": r"^\d{13,19}$"
            },
        )
    )

    # Target Variable Expectations
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_of_type",
            kwargs={"column": target_col, "type_": "int64"},
        )
    )
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={"column": target_col, "value_set": [0, 1]},
        )
    )
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": target_col},
        )
    )

    # Save the expectation suite
    context.add_or_update_expectation_suite(expectation_suite=suite)

    # Run Validation
    data_asset_name = "fraud_data_asset"
    try:
        data_asset = datasource.add_dataframe_asset(name=data_asset_name, dataframe=df)
    except Exception:
        logger.info("Data asset already exists.")
        data_asset = datasource.get_asset(data_asset_name)

    batch_request = data_asset.build_batch_request(dataframe=df)

    checkpoint = gx.checkpoint.SimpleCheckpoint(
        name="fraud_data_checkpoint",
        data_context=context,
        validations=[{
            "batch_request": batch_request,
            "expectation_suite_name": "fraud",
        }],
    )
    checkpoint_result = checkpoint.run()

    # Manual Assertions
    # pd_df_ge = gx.from_pandas(df)
    assert set(df.columns) == set(expected_columns)
    assert df["trans_num"].notnull().all() and df["trans_num"].is_unique
    assert df[target_col].isin([0, 1, 2]).all()
    assert df["amt"].dtype == "float64"

    if not checkpoint_result["success"]:
        failed = get_validation_results(checkpoint_result)
        failed_rows = failed[failed["Success"] == False]
        logger.warning(f"{len(failed_rows)} expectations failed:\n{failed_rows[['Expectation Type', 'Column', 'Unexpected Percent']]}")
    else:
        logger.info("Great Expectations suite passed successfully.")

    # Return Result DataFrame
    df_validation = get_validation_results(checkpoint_result)
    logger.info(f"Validation suite results: {df_validation['Success'].value_counts().to_dict()}")
    logger.info("Fraud data validation completed successfully.")
    
    return df_validation