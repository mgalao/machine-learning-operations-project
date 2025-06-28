import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def filter_test_data_by_state(ana_data: pd.DataFrame, state: str = 'NY') -> pd.DataFrame:
    return ana_data[ana_data["state"] == state].copy()


def evaluate_predictions(df_with_predict: pd.DataFrame, target_column: str = "is_fraud") -> pd.DataFrame:
    y_true = df_with_predict[target_column]
    y_pred = df_with_predict["prediction"]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "macro_precision": precision_score(y_true, y_pred, average="macro"),
        "macro_recall": recall_score(y_true, y_pred, average="macro"),
    }

    result_df = pd.DataFrame([metrics])

    # Save to reporting folder
    result_df.to_csv("data/08_reporting/eval_metric_drift_data.csv", index=False)

    return result_df
