"""Column names and metric calculations shared by training and the Streamlit app."""

from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "messidor_retinopathy.csv"
TEST_PATH = ROOT / "test_data.csv"
MODEL_DIR = ROOT / "model"
METRICS_PATH = MODEL_DIR / "metrics.json"

FEATURE_COLUMNS = [
    "quality_ok",
    "prescreen_abnormal",
    "ma_count_alpha_0_5",
    "ma_count_alpha_0_6",
    "ma_count_alpha_0_7",
    "ma_count_alpha_0_8",
    "ma_count_alpha_0_9",
    "ma_count_alpha_1_0",
    "exudate_norm_1",
    "exudate_norm_2",
    "exudate_norm_3",
    "exudate_norm_4",
    "exudate_norm_5",
    "exudate_norm_6",
    "exudate_norm_7",
    "exudate_norm_8",
    "macula_optic_distance",
    "optic_disc_diameter",
    "am_fm_label",
]

TARGET_COLUMN = "dr_signs"
CLASS_NAMES = ["No DR signs", "DR signs"]
RANDOM_STATE = 42
TEST_SIZE = 0.20

MODEL_CATALOG = [
    {
        "key": "logistic_regression",
        "label": "Logistic Regression",
        "file": "logistic_regression.joblib",
    },
    {
        "key": "decision_tree",
        "label": "Decision Tree",
        "file": "decision_tree.joblib",
    },
    {
        "key": "knn",
        "label": "KNN",
        "file": "knn.joblib",
    },
    {
        "key": "naive_bayes",
        "label": "Naive Bayes",
        "file": "naive_bayes.joblib",
    },
    {
        "key": "svm",
        "label": "Support Vector Machine",
        "file": "svm.joblib",
    },
]

METRIC_KEYS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def positive_scores(model, features):
    """Probability of the positive class (dr_signs = 1)."""
    probabilities = model.predict_proba(features)
    class_list = [int(label) for label in model.classes_]
    positive_index = class_list.index(1)
    return probabilities[:, positive_index]


def score_model(model, features, target):
    """Hold-out metrics. Precision, recall, and F1 use the positive class."""
    predictions = np.asarray(model.predict(features)).astype(int)
    scores = positive_scores(model, features)
    y_true = np.asarray(target).astype(int)
    unique = set(y_true.tolist())
    if unique != {0, 1}:
        raise ValueError(
            "The target column must contain both classes 0 and 1 to score AUC."
        )
    return {
        "Accuracy": float(accuracy_score(y_true, predictions)),
        "AUC": float(roc_auc_score(y_true, scores)),
        "Precision": float(
            precision_score(y_true, predictions, pos_label=1, zero_division=0)
        ),
        "Recall": float(recall_score(y_true, predictions, pos_label=1, zero_division=0)),
        "F1": float(f1_score(y_true, predictions, pos_label=1, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, predictions)),
    }
