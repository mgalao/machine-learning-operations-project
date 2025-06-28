# """
# This is a boilerplate pipeline
# generated using Kedro 0.18.8
# """

import pandas as pd
import numpy as np
import logging

from .utils import calculate_psi, find_optimal_bins_with_knee
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import nannyml as nml

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def data_drift(data_reference: pd.DataFrame, data_analysis: pd.DataFrame) -> tuple:
    numerical_features = ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long']
    categorical_features = ['category', 'gender', 'city', 'state', 'zip']

    # ----------- PSI SECTION with Optimal Bins -----------
    optimal_bins_per_feature, _ = find_optimal_bins_with_knee(
        data_reference[numerical_features],
        data_analysis[numerical_features],
        buckettype='quantiles'
    )

    psi_scores = {}
    for feature in numerical_features:
        optimal_bins = optimal_bins_per_feature.get(feature, 10)
        psi_value = calculate_psi(
            expected=data_reference[feature].values,
            actual=data_analysis[feature].values,
            buckettype='quantiles',
            buckets=optimal_bins,
            axis=0
        )[0]
        psi_scores[feature] = psi_value

    psi_series = pd.Series(psi_scores, index=numerical_features)
    psi_df = pd.DataFrame({
        "feature": psi_series.index,
        "psi_score": psi_series.values
    })

    # Drift detection
    drifted_features = psi_df[psi_df["psi_score"] > 0.1]["feature"].tolist()
    stable_features = psi_df[psi_df["psi_score"] <= 0.1]["feature"].tolist()
    trigger_retraining = len(drifted_features) > 0

    if drifted_features:
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
    threshold = nml.thresholds.ConstantThreshold(lower=None, upper=0.25)
    calc = nml.UnivariateDriftCalculator(
        column_names=categorical_features,
        treat_as_categorical=categorical_features,
        chunk_size=1000,
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

    return (
        psi_df,               # DataFrame with 'feature' and 'psi_score'
        psi_pca,              # Series with PCA PSI
        drift_df,             # NannyML categorical drift output
        drifted_features,     # Features with drift detected
        stable_features,      # Features with no significant drift
        trigger_retraining    # Boolean flag: True if drift detected
    )
