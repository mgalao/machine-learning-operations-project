import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.fraud_project.pipelines.data_drift.nodes import data_drift

@pytest.fixture
def sample_data():
    np.random.seed(42)
    # Sampel Reference data from various U.S. states
    ref_data = pd.DataFrame({
        "amt": np.random.normal(100, 10, 200),
        "lat": np.random.uniform(25.0, 49.0, 200),   # U.S. latitudes
        "long": np.random.uniform(-124.0, -67.0, 200),  # U.S. longitudes
        "city_pop": np.random.randint(10_000, 1_000_000, 200),
        "merch_lat": np.random.uniform(25.0, 49.0, 200),
        "merch_long": np.random.uniform(-124.0, -67.0, 200),
        "merchant": ["fraud_Kilback LLC"] * 200,
        "category": ["gas_transport"] * 200,
        "gender": np.random.choice(["M", "F"], 200),
        "city": np.random.choice(["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"], 200),
        "state": np.random.choice(["NY", "CA", "IL", "TX", "AZ"], 200),
        "zip": np.random.choice(["10001", "90001", "60601", "77001", "85001"], 200),
        "job": ["Film/video editor"] * 200
    })

    # Analysis data: Only New York (NY)
    ana_data = ref_data.copy()
    ana_data["state"] = "NY"
    ana_data["city"] = "New York"
    ana_data["zip"] = "10001"
    ana_data["amt"] += 1500  # introduce drift on amount

    return ref_data, ana_data


def test_data_drift_outputs(sample_data, tmp_path):
    ref_data, ana_data = sample_data

    # Run data drift analysis
    psi_series, psi_pca, drift_df, drifted_features = data_drift(ref_data, ana_data)

    # --- Test PSI results ---
    assert isinstance(psi_series, pd.Series)
    assert not psi_series.empty
    assert all(0 <= x for x in psi_series)

    # --- Test PCA PSI results ---
    assert isinstance(psi_pca, pd.Series)
    assert "PC1" in psi_pca.index and "PC2" in psi_pca.index

    # --- Test NannyML drift output ---
    assert isinstance(drift_df, pd.DataFrame)
    assert not drift_df.empty

    # --- Test drifted features list ---
    assert isinstance(drifted_features, list)
    assert all(isinstance(item, str) for item in drifted_features)

    # --- Test Evidently output file exists ---
    html_path = Path("data/08_reporting/data_drift_evidently.html")
    assert html_path.exists()

    html_plot_path = Path("data/08_reporting/univariate_drift_nml.html")
    assert html_plot_path.exists()
