"""Score saved Messidor classifiers on an uploaded test CSV. Does not retrain."""

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from metrics_utils import (
    CLASS_NAMES,
    FEATURE_COLUMNS,
    METRICS_PATH,
    METRIC_KEYS,
    MODEL_CATALOG,
    MODEL_DIR,
    TARGET_COLUMN,
    TEST_PATH,
    score_model,
)

st.set_page_config(
    page_title="Messidor DR classifier",
    page_icon=":material/visibility:",
    layout="wide",
)


@st.cache_resource
def load_models():
    loaded = {}
    for item in MODEL_CATALOG:
        path = MODEL_DIR / item["file"]
        if not path.exists():
            raise FileNotFoundError(f"Missing saved model: {path.name}")
        loaded[item["label"]] = joblib.load(path)
    return loaded


def read_training_summary():
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text())


def load_default_test():
    if not TEST_PATH.exists():
        return None
    return pd.read_csv(TEST_PATH)


def prepare_upload(frame):
    renamed = frame.copy()
    renamed.columns = [str(column).strip() for column in renamed.columns]
    missing = [column for column in FEATURE_COLUMNS + [TARGET_COLUMN] if column not in renamed.columns]
    if missing:
        raise ValueError(
            "The CSV needs every training feature plus the target column "
            f"`{TARGET_COLUMN}`. Missing: {', '.join(missing)}"
        )
    prepared = renamed[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    for column in FEATURE_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    if prepared[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Every feature column must be numeric, with no missing values.")
    prepared[TARGET_COLUMN] = pd.to_numeric(prepared[TARGET_COLUMN], errors="coerce")
    if prepared[TARGET_COLUMN].isna().any():
        raise ValueError(f"`{TARGET_COLUMN}` must be numeric, with 1 = DR signs and 0 = none.")
    prepared[TARGET_COLUMN] = prepared[TARGET_COLUMN].astype(int)
    if prepared.empty:
        raise ValueError("The CSV has no rows to score.")
    return prepared


def score_every_model(models, features, target):
    rows = []
    for item in MODEL_CATALOG:
        metrics = score_model(models[item["label"]], features, target)
        row = {"ML Model Name": item["label"]}
        row.update({key: round(metrics[key], 4) for key in METRIC_KEYS})
        rows.append(row)
    return pd.DataFrame(rows)


def draw_confusion(target, predictions):
    matrix = confusion_matrix(target, predictions, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(4.4, 3.6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="GnBu",
        cbar=False,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axis,
    )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title("Confusion matrix")
    figure.tight_layout()
    return figure


def draw_comparison(table):
    figure, axis = plt.subplots(figsize=(8.2, 3.3))
    x_positions = range(len(table))
    width = 0.36
    axis.bar(
        [position - width / 2 for position in x_positions],
        table["AUC"],
        width=width,
        label="AUC",
        color="#0F6E6E",
    )
    axis.bar(
        [position + width / 2 for position in x_positions],
        table["F1"],
        width=width,
        label="F1 (DR signs)",
        color="#D08C60",
    )
    axis.set_xticks(list(x_positions))
    axis.set_xticklabels(table["ML Model Name"], rotation=15, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.legend(frameon=False)
    axis.set_title("AUC and positive-class F1 on this file")
    figure.tight_layout()
    return figure


def main():
    st.markdown(
        """
        <div style="font-size:0.82rem;letter-spacing:0.04em;color:#0F6E6E;font-weight:650;">
        BITS WILP · M.Tech AIML · Machine Learning Assignment 2
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.title("Messidor diabetic-retinopathy sign classifier")
    st.write(
        "Five models were trained on UCI Messidor image measurements and saved "
        "to disk. This page loads those files and scores the CSV you provide. "
        "It does not fit the models again."
    )
    st.caption(
        "Academic demo only. These scores are not a medical diagnosis. "
        "Positive class = 1 (image has signs of diabetic retinopathy)."
    )

    try:
        models = load_models()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    with st.sidebar:
        st.header("Dataset")
        st.write(
            "UCI Diabetic Retinopathy Debrecen. Each row is one Messidor fundus "
            "image described by 19 numeric measurements: quality, pre-screening, "
            "six microaneurysm counts, eight normalized exudate values, macula–optic "
            "disc distance, optic-disc diameter, and an AM/FM label."
        )
        summary = read_training_summary()
        if summary:
            st.write(
                f"Training used {summary['split']['n_train']} rows. "
                f"The bundled test file has {summary['split']['n_test']} rows. "
                f"Winner on that split by AUC: **{summary['winner_by_test_auc']}**."
            )
        st.header("What gets scored")
        st.write(
            "Accuracy, ROC AUC, precision, recall, F1, and Matthews correlation. "
            "Precision, recall, and F1 are for DR signs present."
        )

    uploaded = st.file_uploader(
        "Upload a test CSV with the 19 features and the dr_signs column",
        type=["csv"],
        help="Use test_data.csv from this project, or another file with the same columns.",
    )
    if uploaded is None:
        frame = load_default_test()
        source_name = TEST_PATH.name
        if frame is None:
            st.warning("Add test_data.csv beside app.py, or upload a CSV.")
            st.stop()
        st.info(f"No file uploaded. Showing scores for the bundled `{source_name}`.")
    else:
        frame = pd.read_csv(uploaded)
        source_name = uploaded.name

    try:
        prepared = prepare_upload(frame)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    features = prepared[FEATURE_COLUMNS]
    target = prepared[TARGET_COLUMN]
    positive_rate = float(target.mean())
    st.write(
        f"**{source_name}** · {len(prepared)} rows · "
        f"{positive_rate:.1%} labeled as DR signs"
    )

    labels = [item["label"] for item in MODEL_CATALOG]
    selected = st.selectbox("Choose a classification model", labels, index=0)

    try:
        comparison = score_every_model(models, features, target)
        selected_metrics = score_model(models[selected], features, target)
        predictions = models[selected].predict(features)
    except Exception as error:
        st.error(f"Could not score this file: {error}")
        st.stop()

    st.success("Scored with the saved model. No retraining on this file.")
    cards = st.columns(6)
    for card, key in zip(cards, METRIC_KEYS):
        card.metric(key, f"{selected_metrics[key]:.4f}")

    st.subheader("All five models on this file")
    st.dataframe(comparison, hide_index=True, width="stretch")
    chart = draw_comparison(comparison)
    st.pyplot(chart)
    plt.close(chart)

    detail_left, detail_right = st.columns(2)
    with detail_left:
        st.subheader(f"{selected}: confusion matrix")
        matrix_figure = draw_confusion(target, predictions)
        st.pyplot(matrix_figure)
        plt.close(matrix_figure)
    with detail_right:
        st.subheader(f"{selected}: classification report")
        report = classification_report(
            target,
            predictions,
            labels=[0, 1],
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
        st.dataframe(pd.DataFrame(report).T.round(4), width="stretch")

    with st.expander("Preview of the rows that were scored"):
        st.dataframe(prepared.head(12), hide_index=True, width="stretch")


if __name__ == "__main__":
    main()
else:
    main()
