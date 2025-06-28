"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def stratified_split_by_target(df, target_column: str):
    ref_data, ana_data = train_test_split(
        df,
        test_size=0.2,
        stratify=df[target_column],
        random_state=200,
    )
    return ref_data, ana_data