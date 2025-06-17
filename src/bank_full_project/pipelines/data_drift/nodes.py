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

from .utils import calculate_psi
import matplotlib.pyplot as plt  
import nannyml as nml
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


logger = logging.getLogger(__name__)



def data_drift(data_reference: pd.DataFrame, data_analysis: pd.DataFrame):
    

    #define the threshold for the test as parameters in the parameters catalog
    constant_threshold = nml.thresholds.ConstantThreshold(lower=None, upper=0.2)
    constant_threshold.thresholds(data_reference)

    # Let's initialize the object that will perform the Univariate Drift calculations
    univariate_calculator = nml.UnivariateDriftCalculator(
    column_names=["education"],
    treat_as_categorical=['education'],
    chunk_size=50,
    categorical_methods=['jensen_shannon'],
    thresholds={"jensen_shannon":constant_threshold})

    univariate_calculator.fit(data_reference)
    results = univariate_calculator.calculate(data_analysis).filter(period='analysis', column_names=['education'],methods=['jensen_shannon']).to_df()

    figure = univariate_calculator.calculate(data_analysis).filter(period='analysis', column_names=['education'],methods=['jensen_shannon']).plot(kind='drift')
    figure.write_html("data/08_reporting/univariate_nml.html")
    #generate a report for some numeric features using KS test and evidently ai
    data_drift_report = Report(metrics=[
    DataDriftPreset(cat_stattest='ks', stattest_threshold=0.05)])

    data_drift_report.run(current_data=data_analysis[["age","balance"]] , reference_data=data_reference[["age","balance"]], column_mapping=None)
    data_drift_report.save_html("data/08_reporting/data_drift_report.html")
    
    return results