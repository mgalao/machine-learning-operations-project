"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from great_expectations.core import ExpectationSuite, ExpectationConfiguration


from pathlib import Path

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

conf_path = Path(settings.CONF_SOURCE)
# conf_path = Path('../conf').resolve() # In notebooks, the path is relative to the project root
conf_loader = OmegaConfigLoader(conf_source=conf_path)
credentials = conf_loader["credentials"]

logger = logging.getLogger(__name__)

def validate_schema(df: pd.DataFrame, expected_columns: list, feature_group_name: str):
    """ Validates the schema of a DataFrame against expected columns.
    Args:
        df (pd.DataFrame): The DataFrame to validate.
        expected_columns (list): List of expected column names.
        feature_group_name (str): Name of the feature group for logging.
    Raises:
        ValueError: If the DataFrame does not match the expected schema.
    """
    actual_cols = set(df.columns)
    expected_cols = set(expected_columns)

    missing_cols = expected_cols - actual_cols
    extra_cols = actual_cols - expected_cols

    # Log the schema validation results
    if missing_cols:
        logger.error(f"Missing columns in {feature_group_name}: {missing_cols}")
        raise ValueError(f"Missing columns in {feature_group_name}: {missing_cols}")
    if extra_cols:
        logger.error(f"Unexpected extra columns in {feature_group_name}: {extra_cols}")
        raise ValueError(f"Unexpected extra columns in {feature_group_name}: {extra_cols}")
    
    logger.info(f"[{feature_group_name}] Schema validation passed.")

def build_expectation_suite(
    expectation_suite_name: str, 
    feature_group: str,
    expected_numeric_columns: list = None,
    expected_categorical_columns: list = None,
    target_col: str = None
) -> ExpectationSuite:
    """
    Builds a Great Expectations ExpectationSuite with validations tailored to a given feature group.

    Args:
        expectation_suite_name (str): Name of the expectation suite.
        feature_group (str): One of ['numerical_features', 'categorical_features', 'target'].

    Returns:
        ExpectationSuite: A Great Expectations suite with configured expectations.
    """
    logger.info(f"Building expectation suite for: {feature_group}")
    # Create an empty ExpectationSuite
    expectation_suite = ExpectationSuite(expectation_suite_name=expectation_suite_name)
    
    # Add expectations based on the feature group

    # Numerical features
    if feature_group == 'numerical_features':
        # Expect columns to be correct type and mostly not null
        numerical_columns = [col for col in expected_numeric_columns if col not in ['trans_num', 'datetime']]

        for col in numerical_columns:
            int_columns = ['age', 'city_pop', 'unix_time']
            expected_type = "int64" if col in int_columns else "float64"
            
            expectation_suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_of_type",
                    kwargs={"column": col, "type_": expected_type},
                )
            )
            expectation_suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_not_be_null",
                    kwargs={"column": col, "mostly": 0.95},
                )
            )

        # Specific expectation for 'datetime'
        expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_of_type",
                kwargs={"column": "datetime", "type_": "datetime64[ns]"},
            )
        )

        # Specific expectation for 'age'
        expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "age", "min_value": 18, "max_value": 120, "mostly": 0.95},
            )
        )

        # Specific expectations for 'amt'
        expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_min_to_be_between",
                kwargs={"column": "amt", "min_value": 0, "strict_min": True},
            )
        )
        expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_max_to_be_between",
                kwargs={"column": "amt", "max_value": 100000, "strict_max": False},
            )
        )

        # Add expectations for coordinates
        coordinate_expectations = {
            "lat": (-90, 90),
            "long": (-180, 180),
            "merch_lat": (-90, 90),
            "merch_long": (-180, 180),
        }
        for col, (min_val, max_val) in coordinate_expectations.items():
            expectation_suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_between",
                    kwargs={"column": col, "min_value": min_val, "max_value": max_val},
                )
            )

    # Categorical features
    if feature_group == 'categorical_features':
        # Expect columns to be correct type and mostly not null
        categorical_columns = [col for col in expected_categorical_columns if col not in ['trans_num', 'datetime']]

        for col in categorical_columns:
            expectation_suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_of_type",
                    kwargs={"column": col, "type_": "object"},
                )
            )
            expectation_suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_not_be_null",
                    kwargs={"column": col, "mostly": 0.95},
                )
            )

        # Allowed categories for 'gender'
        expected_genders = ["M", "F", "U"]
        expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_distinct_values_to_be_in_set",
                kwargs={"column": "gender", "value_set": expected_genders},
            )
        )

        # Allowed categories for 'state'
        expected_states = [
            'MO', 'CA', 'OR', 'PA', 'VA', 'NE', 'IA', 'IL', 'WV', 'OK', 'GA',
            'MN', 'NM', 'NY', 'NC', 'LA', 'ME', 'KS', 'NV', 'NJ', 'AL', 'TX',
            'SD', 'MA', 'MI', 'SC', 'MT', 'FL', 'MS', 'IN', 'UT', 'CT', 'KY',
            'OH', 'MD', 'ID', 'ND', 'TN', 'WI', 'VT', 'WY', 'CO', 'NH', 'AR',
            'WA', 'AZ', 'HI', 'DC', 'RI', 'AK', 'DE'
        ]
        expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_distinct_values_to_be_in_set",
                kwargs={"column": "state","value_set": expected_states},
            )
        )

    # Target feature
    if feature_group == 'target':
        # Expect 'is_fraud' to be of type int64 and have values 0 or 1
        expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_of_type",
                kwargs={"column": target_col, "type_": "int64"},
            )
        )
        expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_distinct_values_to_be_in_set",
                kwargs={"column": target_col, "value_set": [0, 1]},
            )
        )
     
    return expectation_suite

import hopsworks

def to_feature_store(
    data: pd.DataFrame,
    group_name: str,
    feature_group_version: int,
    description: str,
    group_description: dict,
    validation_expectation_suite: ExpectationSuite,
    credentials_input: dict
):
    """
    This function takes in a pandas DataFrame and a validation expectation suite,
    performs validation on the data using the suite, and then saves the data to a
    feature store in the feature store.

    Args:
        data (pd.DataFrame): Dataframe with the data to be stored
        group_name (str): Name of the feature group.
        feature_group_version (int): Version of the feature group.
        description (str): Description for the feature group.
        group_description (List[Dict[str, str]]): Description of each feature in the feature group.
        validation_expectation_suite (ExpectationSuite): group of expectations to check data.
        credentials_input (dict): Dictionary with the credentials to connect to the project.        
    Returns:
        A dictionary with the feature view version, feature view name and training dataset feature version.
    """
    logger.info(f"Pushing '{group_name}' to feature store...")
    
    # Connect to feature store
    project = hopsworks.login(
        api_key_value=credentials_input["FS_API_KEY"],
        project=credentials_input["FS_PROJECT_NAME"]
    )
    feature_store = project.get_feature_store()

    # Create feature group
    object_feature_group = feature_store.get_or_create_feature_group(
        name=group_name,
        version=feature_group_version,
        description= description,
        primary_key=["trans_num"],
        event_time="datetime",
        online_enabled=False,
        expectation_suite=validation_expectation_suite,
    )

    # Insert data
    object_feature_group.insert(
        features=data,
        overwrite=False,
        write_options={"wait_for_job": True},
    )

    logger.info(f"Inserted {len(data)} rows into feature group: {group_name}")

    # Add feature descriptions
    for description in group_description:
        object_feature_group.update_feature_description(
            description["name"], description["description"]
        )

    # Update statistics
    object_feature_group.statistics_config = {
        "enabled": True,
        "histograms": True,
        "correlations": True,
    }
    object_feature_group.update_statistics_config()
    object_feature_group.compute_statistics()

def ingestion(df: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    """
    Cleans and uploads credit card transaction data to the feature store, split by feature group.

    Args:
        df (pd.DataFrame): Input dataset containing transactions and labels.
        parameters (Dict[str, Any]): Configuration dict, including target column and flags.

    Returns:
        pd.DataFrame: The cleaned and processed full dataset.
    """
    logger.info("Starting data ingestion pipeline...")

    # Load the dataset
    df = df.copy().drop_duplicates().reset_index(drop=True)
    logger.info(f"Dataset loaded with shape: {df.shape}")
    
    # Fix datetime column
    df["datetime"] = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
    df.drop(columns=["trans_date_trans_time"], inplace=True)
    df.drop(columns=["unix_time"], inplace=True) # Drop unix_time as it is redundant

    logger.info("Created 'datetime' column from 'trans_date_trans_time'.")

    # Convert IDs and categorical numerics to string
    id_columns = ["cc_num", "zip", "merch_zipcode"]
    for col in id_columns:
        # 'Int64' if the column may contain nulls
        if df[col].isnull().any():
            df[col] = df[col].astype("Int64").astype(str)
        else:
            df[col] = df[col].astype(str)

    logger.info("Parsed successfully categorical columns to string.")

    # Create age feature from dob and drop dob
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
    df["age"] = (df["datetime"] - df["dob"]).dt.days // 365
    df.drop(columns=["dob"], inplace=True)

    logger.info("Created 'age' feature from 'dob'.")

    # Identify feature types
    numerical_features = df.select_dtypes(include=["int64", "float64", "Int64"]).columns.tolist()
    categorical_features = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    
    # Remove the target column from the corresponding feature list
    target_col = parameters["target_column"]
    if target_col in numerical_features: numerical_features.remove(target_col)
    if target_col in categorical_features: categorical_features.remove(target_col)

    # Ensure consistent object dtype for categoricals
    df[categorical_features] = df[categorical_features].astype("object")

    # Define feature groups
    numeric_cols = list(dict.fromkeys(["trans_num", "datetime"] + numerical_features))
    df_numeric = df[numeric_cols]

    cat_cols = list(dict.fromkeys(["trans_num", "datetime"] + categorical_features))
    df_categorical = df[cat_cols]

    df_target = df[["trans_num", "datetime", target_col]]

    # Expected columns
    expected_numeric_columns = ['trans_num', 'datetime', 'age', 'amt', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long']
    expected_categorical_columns = ['trans_num', 'datetime', 'cc_num', 'merchant', 'category', 'first', 'last', 'gender', 'street', 'city', 'state', 'zip', 'job', 'merch_zipcode']
    expected_target_columns = ['trans_num', 'datetime', target_col]

    # Validate schemas
    try:
        validate_schema(df_numeric, expected_numeric_columns, "numerical_features")
        validate_schema(df_categorical, expected_categorical_columns, "categorical_features")
        validate_schema(df_target, expected_target_columns, "target_features")
    except ValueError as e:
        logger.exception("Schema validation failed.")
        raise

    logger.info("All schema validations passed.")

    # Expectation suites
    suite_num = build_expectation_suite(
        "numerical_expectations",
        "numerical_features",
        expected_numeric_columns=expected_numeric_columns,
    )

    suite_cat = build_expectation_suite(
        "categorical_expectations",
        "categorical_features",
        expected_categorical_columns=expected_categorical_columns,
    )

    suite_target = build_expectation_suite(
        "target_expectations",
        "target",
        target_col=target_col,
    )

    # Feature descriptions
    numerical_feature_descriptions = [
        {"name": "age", "description": "Age of the cardholder calculated from dob and transaction date."},
        {"name": "amt", "description": "Amount of the transaction."},
        {"name": "lat", "description": "Latitude coordinate of the transaction."},
        {"name": "long", "description": "Longitude coordinate of the transaction."},
        {"name": "city_pop", "description": "Population of the city where the transaction occurred."},
        {"name": "unix_time", "description": "Unix timestamp of the transaction."},
        {"name": "merch_lat", "description": "Latitude coordinate of the merchant."},
        {"name": "merch_long", "description": "Longitude coordinate of the merchant."},
        {"name": "trans_num", "description": "Unique transaction number."},
        {"name": "datetime", "description": "Parsed datetime of the transaction from trans_date_trans_time."}
    ]

    categorical_feature_descriptions = [
        {"name": "cc_num", "description": "Credit card number as string."},
        {"name": "merchant", "description": "Merchant or store where the transaction occurred."},
        {"name": "category", "description": "Merchant category where the transaction occurred."},
        {"name": "first", "description": "First name of the cardholder."},
        {"name": "last", "description": "Last name of the cardholder."},
        {"name": "gender", "description": "Gender of the cardholder."},
        {"name": "street", "description": "Street address of the cardholder."},
        {"name": "city", "description": "Name of the city where the cardholder resides."},
        {"name": "state", "description": "Name of the state where the cardholder resides."},
        {"name": "zip", "description": "Zip code of the cardholder."},
        {"name": "job", "description": "Occupation of the cardholder."},
        {"name": "merch_zipcode", "description": "Zip code of the merchant."},
        {"name": "trans_num", "description": "Unique transaction number."},
        {"name": "datetime", "description": "Parsed datetime of the transaction from trans_date_trans_time."}
    ]

    target_feature_descriptions = [
        {"name": "is_fraud", "description": "Indicator of whether the transaction is fraudulent (0 = Not Fraud, 1 = Fraud)."},
        {"name": "trans_num", "description": "Unique transaction number."},
        {"name": "datetime", "description": "Parsed datetime of the transaction from trans_date_trans_time."}
    ]

    # Push to Hopsworks
    if parameters.get("to_feature_store", False):
        logger.info("Uploading to Hopsworks feature store...")

        to_feature_store(
            df_numeric,
            group_name="numerical_features",
            feature_group_version=1,
            description="Numerical features",
            group_description=numerical_feature_descriptions,
            validation_expectation_suite=suite_num,
            credentials_input=credentials["feature_store"],
        )

        to_feature_store(
            df_categorical,
            group_name="categorical_features",
            feature_group_version=1,
            description="Categorical features",
            group_description=categorical_feature_descriptions,
            validation_expectation_suite=suite_cat,
            credentials_input=credentials["feature_store"],
        )

        to_feature_store(
            df_target,
            group_name="target_features",
            feature_group_version=1,
            description="Target features",
            group_description=target_feature_descriptions,
            validation_expectation_suite=suite_target,
            credentials_input=credentials["feature_store"],
        )
        
        logger.info("Data uploaded to Hopsworks successfully.")

    return df