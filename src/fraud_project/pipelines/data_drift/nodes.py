# """
# This is a boilerplate pipeline
# generated using Kedro 0.18.8
# """

import pandas as pd
import numpy as np
import logging

from .utils import calculate_psi
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import nannyml as nml

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

def data_drift(data_reference: pd.DataFrame, data_analysis: pd.DataFrame) -> dict:
    numerical_features = ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long']
    categorical_features = ['merchant', 'category', 'gender', 'city', 'state', 'zip', 'job']

    # ----------- PSI SECTION -----------
    psi_scores = calculate_psi(
        expected=data_reference[numerical_features].values,
        actual=data_analysis[numerical_features].values,
        buckettype='quantiles',
        buckets=10,
        axis=0
    )

    psi_series = pd.Series(psi_scores, index=numerical_features)
    drifted_features = psi_series[psi_series > 0.1]

    if not drifted_features.empty:
        logger.warning("Significant data drift detected via PSI:\n%s", drifted_features)

    # ----------- EVIDENTLY REPORT -----------
    report = Report(metrics=[DataDriftPreset(cat_stattest='ks', stattest_threshold=0.05)])
    report.run(
        reference_data=data_reference[numerical_features],
        current_data=data_analysis[numerical_features],
        column_mapping=None
    )
    report.save_html("data/08_reporting/data_drift_evidently.html")

    # ----------- NANNYML CATEGORICAL DRIFT -----------
    threshold = nml.thresholds.ConstantThreshold(lower=None, upper=0.2)
    calc = nml.UnivariateDriftCalculator(
        column_names=categorical_features,
        treat_as_categorical=categorical_features,
        chunk_size=50,
        categorical_methods=['jensen_shannon'],
        thresholds={"jensen_shannon": threshold}
    )

    calc.fit(data_reference)
    result = calc.calculate(data_analysis)
    drift_df = result.filter(period='analysis').to_df()

    plot = result.filter(period='analysis').plot(kind='drift')
    plot.write_html("data/08_reporting/univariate_drift_nml.html")

    # ----------- PCA PSI (Optional but Informative) -----------
    scaler = StandardScaler()
    ref_scaled = scaler.fit_transform(data_reference[numerical_features])
    ana_scaled = scaler.transform(data_analysis[numerical_features])

    pca = PCA(n_components=2)
    ref_pca = pca.fit_transform(ref_scaled)
    ana_pca = pca.transform(ana_scaled)

    psi_pca_scores = calculate_psi(ref_pca, ana_pca, buckettype='quantiles')
    psi_pca = pd.Series(psi_pca_scores, index=["PC1", "PC2"])

    if (psi_pca > 0.1).any():
        logger.warning("Significant drift in PCA components detected: \n%s", psi_pca[psi_pca > 0.1])

    return {
        "psi_scores": psi_series.to_dict(),
        "pca_psi_scores": psi_pca.to_dict(),
        "nannyml_drift": drift_df.to_dict(orient="records"),
        "drifted_features": drifted_features.index.tolist()
    }
