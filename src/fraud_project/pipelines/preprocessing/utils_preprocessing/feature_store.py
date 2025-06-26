# import logging
# import pandas as pd
# import hopsworks
# from great_expectations.core import ExpectationSuite

# logger = logging.getLogger(__name__)

# def load_from_feature_store(
#     group_name: str,
#     version: int,
#     credentials: dict
# ) -> pd.DataFrame:
#     """
#     Loads a feature group from Hopsworks feature store.
#     """
#     project = hopsworks.login(
#         api_key_value=credentials["FS_API_KEY"],
#         project=credentials["FS_PROJECT_NAME"]
#     )
#     feature_store = project.get_feature_store()
#     fg = feature_store.get_feature_group(name=group_name, version=version)
#     df = fg.read()
#     logger.info(f"Loaded {len(df)} rows from feature group '{group_name}@{version}'")
#     return df

# def upload_to_feature_store(
#     df: pd.DataFrame,
#     group_name: str,
#     feature_group_version: int,
#     description: str,
#     group_description: list,
#     validation_suite: ExpectationSuite,
#     credentials: dict
# ) -> None:
#     """
#     Uploads a DataFrame to Hopsworks feature store.
#     Intended for use by ingestion and preprocessing pipelines.
#     """
#     logger.info(f"Uploading '{group_name}' to feature store (version {feature_group_version})...")
    
#     project = hopsworks.login(
#         api_key_value=credentials["FS_API_KEY"],
#         project=credentials["FS_PROJECT_NAME"]
#     )
#     feature_store = project.get_feature_store()

#     fg = feature_store.get_or_create_feature_group(
#         name=group_name,
#         version=feature_group_version,
#         description=description,
#         primary_key=["trans_num"],
#         event_time="datetime",
#         online_enabled=False,
#         expectation_suite=validation_suite,
#     )

#     fg.insert(df, overwrite=False, write_options={"wait_for_job": True})

#     for feat in group_description:
#         fg.update_feature_description(feat["name"], feat["description"])

#     fg.statistics_config = {
#         "enabled": True,
#         "histograms": True,
#         "correlations": True,
#     }
#     fg.update_statistics_config()
#     fg.compute_statistics()

#     logger.info(f"Finished uploading '{group_name}'. Rows inserted: {len(df)}")