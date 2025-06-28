import logging
import pandas as pd
import hopsworks
from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings
from great_expectations.core import ExpectationSuite
from pathlib import Path

logger = logging.getLogger(__name__)

def get_credentials():
    conf_path = str(Path('') / settings.CONF_SOURCE)
    conf_loader = OmegaConfigLoader(conf_source=conf_path)
    return conf_loader["credentials"]

def load_from_feature_store(
    group_name: str,
    version: int,
) -> pd.DataFrame:
    """
    Loads a feature group from Hopsworks feature store.
    """
    credentials = get_credentials()
    project = hopsworks.login(
        api_key_value=credentials["feature_store"]["FS_API_KEY"],
        project=credentials["feature_store"]["FS_PROJECT_NAME"]
    )
    feature_store = project.get_feature_store()
    fg = feature_store.get_feature_group(name=group_name, version=version)
    df = fg.read()
    logger.info(f"Loaded {len(df)} rows from feature group '{group_name}@{version}'")
    return df

def upload_to_feature_store(
    df: pd.DataFrame,
    group_name: str,
    feature_group_version: int,
    description: str,
    group_description: list,
    primary_key: list,
    event_time: str
) -> None:
    """
    Uploads a DataFrame to Hopsworks feature store (no Great Expectations validation).

    Args:
        df (pd.DataFrame): DataFrame to upload.
        group_name (str): Feature group name.
        feature_group_version (int): Version of the feature group.
        description (str): Description of the feature group.
        group_description (list): List of feature metadata dicts, e.g. [{"name": ..., "description": ...}, ...].
        credentials (dict): Hopsworks API credentials with keys "FS_API_KEY" and "FS_PROJECT_NAME".
        primary_key (list): List of column names to set as primary key.
        event_time (str): Column name for event timestamp.
    """
    logger.info(f"Uploading '{group_name}' to feature store (version {feature_group_version})...")
    try:
        credentials = get_credentials()
        project = hopsworks.login(
            api_key_value=credentials["feature_store"]["FS_API_KEY"],
            project=credentials["feature_store"]["FS_PROJECT_NAME"]
        )
        feature_store = project.get_feature_store()

        fg = feature_store.get_or_create_feature_group(
            name=group_name,
            version=feature_group_version,
            description=description,
            primary_key=primary_key,
            event_time=event_time,
            online_enabled=False
        )

        fg.insert(df, overwrite=False, write_options={"wait_for_job": True})

        for feat in group_description:
            fg.update_feature_description(feat["name"], feat["description"])

        fg.statistics_config = {
            "enabled": True,
            "histograms": True,
            "correlations": True,
        }
        fg.update_statistics_config()
        fg.compute_statistics()

        logger.info(f"Finished uploading '{group_name}'. Rows inserted: {len(df)}")

    except Exception as e:
        logger.error(f"Failed to upload '{group_name}': {e}")
        raise

def upload_multiple_feature_groups(
    dfs: dict,
    feature_groups_config: dict,
):
    """
    Uploads multiple DataFrames to their respective feature groups.

    Args:
        dfs: Dict[str, pd.DataFrame], keys like 'numerical', 'categorical', 'target'
        feature_groups_config: Dict with name, version, description per group
    """
    for key, df in dfs.items():
        if key not in feature_groups_config:
            logger.warning(f"Feature group config for '{key}' not found, skipping upload.")
            continue
        fg_config = feature_groups_config[key]
        upload_to_feature_store(
            df=df,
            group_name=fg_config["name"],
            feature_group_version=fg_config["version"],
            description=fg_config["description"],
            group_description=[],
            primary_key=["trans_num"],
            event_time="datetime"
        )