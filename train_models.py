"""Train the five required classifiers and save them for the Streamlit app.

The held-out CSV written here is the file the app scores. Models are fit on
the training rows only.
"""

import json
import warnings
from time import perf_counter

warnings.filterwarnings(
    "ignore",
    message="The `probability` parameter was deprecated",
    category=FutureWarning,
)

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from metrics_utils import (
    DATA_PATH,
    FEATURE_COLUMNS,
    METRICS_PATH,
    MODEL_CATALOG,
    MODEL_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_PATH,
    TEST_SIZE,
    score_model,
)


def load_messidor(path=DATA_PATH):
    frame = pd.read_csv(path)
    missing = [name for name in FEATURE_COLUMNS + [TARGET_COLUMN] if name not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    frame = frame[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype(int)
    if frame[TARGET_COLUMN].nunique() != 2:
        raise ValueError("dr_signs must be binary.")
    if len(frame) < 500:
        raise ValueError("Assignment needs at least 500 instances.")
    if len(FEATURE_COLUMNS) < 12:
        raise ValueError("Assignment needs at least 12 features.")
    return frame


def _search(estimator, grid, X_train, y_train, cv):
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )
    search = GridSearchCV(
        pipeline,
        grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def _fit_svm(best_params, X_train, y_train):
    """Refit the chosen SVM with probability estimates for the app."""
    model_params = {
        key.split("model__", 1)[1]: value
        for key, value in best_params.items()
        if key.startswith("model__")
    }
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                SVC(
                    kernel="rbf",
                    probability=True,
                    random_state=RANDOM_STATE,
                    **model_params,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def train_all(frame):
    features = frame[FEATURE_COLUMNS]
    target = frame[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    specs = [
        (
            "logistic_regression",
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            {
                "model__C": [0.1, 1.0, 10.0],
                "model__class_weight": [None, "balanced"],
            },
            False,
        ),
        (
            "decision_tree",
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {
                "model__max_depth": [4, 8, 12],
                "model__min_samples_leaf": [2, 8],
                "model__class_weight": [None, "balanced"],
            },
            False,
        ),
        (
            "knn",
            KNeighborsClassifier(),
            {
                "model__n_neighbors": [3, 7, 15],
                "model__weights": ["uniform", "distance"],
            },
            False,
        ),
        (
            "naive_bayes",
            GaussianNB(),
            {"model__var_smoothing": [1e-9, 1e-7, 1e-5]},
            False,
        ),
        (
            "svm",
            SVC(kernel="rbf", random_state=RANDOM_STATE),
            {
                "model__C": [0.5, 2.0, 8.0],
                "model__gamma": ["scale"],
                "model__class_weight": [None, "balanced"],
            },
            True,
        ),
    ]

    fitted = {}
    reports = []
    for key, estimator, grid, is_svm in specs:
        started = perf_counter()
        search = _search(estimator, grid, X_train, y_train, cv)
        if is_svm:
            model = _fit_svm(search.best_params_, X_train, y_train)
        else:
            model = search.best_estimator_
        elapsed = perf_counter() - started
        metrics = score_model(model, X_test, y_test)
        fitted[key] = model
        reports.append(
            {
                "key": key,
                "label": next(item["label"] for item in MODEL_CATALOG if item["key"] == key),
                "best_params": search.best_params_,
                "cv_auc": float(search.best_score_),
                "fit_seconds": round(elapsed, 2),
                "test_metrics": metrics,
            }
        )
        print(f"fitted {key:22} cv_auc={search.best_score_:.4f}  {elapsed:.1f}s")

    return fitted, reports, X_train, X_test, y_train, y_test


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _logistic_drivers(model):
    coefficients = model.named_steps["model"].coef_[0]
    order = np.argsort(np.abs(coefficients))[::-1]
    return [
        {"feature": FEATURE_COLUMNS[index], "scaled_coefficient": float(coefficients[index])}
        for index in order[:6]
    ]


def _tree_drivers(model):
    importances = model.named_steps["model"].feature_importances_
    order = np.argsort(importances)[::-1]
    return [
        {"feature": FEATURE_COLUMNS[index], "importance": float(importances[index])}
        for index in order[:6]
    ]


def _print_table(reports):
    header = f"{'ML Model Name':<24} {'Accuracy':>8} {'AUC':>8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'MCC':>8}"
    print("\n" + header)
    print("-" * len(header))
    for report in reports:
        metrics = report["test_metrics"]
        print(
            f"{report['label']:<24} "
            f"{metrics['Accuracy']:8.4f} "
            f"{metrics['AUC']:8.4f} "
            f"{metrics['Precision']:10.4f} "
            f"{metrics['Recall']:8.4f} "
            f"{metrics['F1']:8.4f} "
            f"{metrics['MCC']:8.4f}"
        )


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. The UCI Messidor feature table is stored in data/."
        )
    frame = load_messidor()
    print(f"rows={len(frame)} features={len(FEATURE_COLUMNS)}")
    print(frame[TARGET_COLUMN].value_counts().sort_index().to_string())
    ma_corr = frame["ma_count_alpha_0_5"].corr(frame["ma_count_alpha_1_0"])
    print(f"corr(ma_count_alpha_0_5, ma_count_alpha_1_0)={ma_corr:.3f}")

    fitted, reports, X_train, X_test, y_train, y_test = train_all(frame)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for item in MODEL_CATALOG:
        path = MODEL_DIR / item["file"]
        joblib.dump(fitted[item["key"]], path)
        reloaded = joblib.load(path)
        reloaded_scores = score_model(reloaded, X_test, y_test)
        original_scores = next(
            report["test_metrics"] for report in reports if report["key"] == item["key"]
        )
        for metric_name, value in original_scores.items():
            if abs(reloaded_scores[metric_name] - value) > 1e-9:
                raise RuntimeError(f"Reload mismatch for {item['key']} {metric_name}")

    test_frame = X_test.copy()
    test_frame[TARGET_COLUMN] = y_test.to_numpy()
    test_frame.to_csv(TEST_PATH, index=False)

    majority = int(y_train.mode().iloc[0])
    majority_accuracy = float((y_test.to_numpy() == majority).mean())
    winner = max(reports, key=lambda report: report["test_metrics"]["AUC"])

    payload = {
        "dataset": {
            "name": "Diabetic Retinopathy Debrecen (Messidor features)",
            "source": "UCI Machine Learning Repository, dataset 329",
            "rows": int(len(frame)),
            "features": len(FEATURE_COLUMNS),
            "class_counts": {
                str(label): int(count)
                for label, count in frame[TARGET_COLUMN].value_counts().sort_index().items()
            },
            "ma_count_correlation_0_5_vs_1_0": round(float(ma_corr), 4),
        },
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratified": True,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "majority_class_on_train": majority,
            "majority_accuracy_on_test": round(majority_accuracy, 4),
        },
        "positive_class": 1,
        "metric_definition": (
            "Accuracy and MCC use both classes. AUC uses the predicted probability "
            "of class 1. Precision, recall, and F1 are for class 1 (DR signs present)."
        ),
        "winner_by_test_auc": winner["label"],
        "logistic_top_scaled_coefficients": _logistic_drivers(fitted["logistic_regression"]),
        "tree_top_importances": _tree_drivers(fitted["decision_tree"]),
        "models": reports,
    }
    METRICS_PATH.write_text(json.dumps(_jsonable(payload), indent=2) + "\n")
    _print_table(reports)
    print(f"\nMajority-class accuracy on test: {majority_accuracy:.4f}")
    print(f"Winner by test AUC: {winner['label']}")
    print(f"Saved models in {MODEL_DIR}")
    print(f"Saved test rows in {TEST_PATH} ({len(test_frame)} rows)")
    return payload


if __name__ == "__main__":
    main()
