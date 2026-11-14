"""
MODEL TRAINING AND EVALUATION
Run: python train_model.py
The script cleans the dataset, performs a stratified 80:20 split,
trains TF-IDF + Multinomial Naive Bayes, evaluates it and saves outputs.
"""

from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from src_data_loader import load_and_clean_dataset

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = MODEL_DIR / "sms_spam_pipeline.joblib"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
REPORT_PATH = OUTPUT_DIR / "classification_report.txt"
MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"

RANDOM_STATE = 42
TEST_SIZE = 0.20

def main():
    """Execute the complete reproducible machine-learning workflow."""
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Load the genuine dataset and clean invalid/duplicate rows.
    dataset = load_and_clean_dataset()
    X = dataset["message"]
    y = dataset["label"]

    # 2. Split before fitting TF-IDF to prevent test-data vocabulary leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 3. A single Pipeline ensures the same text transformation at training
    #    and prediction time. Stop words are deliberately retained because
    #    words such as "not" can change a message's meaning.
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.98,
            sublinear_tf=True,
        )),
        ("classifier", MultinomialNB(alpha=0.5)),
    ])

    # 4. Learn vocabulary/IDF weights from training messages and fit Naive Bayes.
    model.fit(X_train, y_train)

    # 5. Predict labels for messages never used during model fitting.
    predictions = model.predict(X_test)

    # 6. Calculate all metrics required by the lab instructions.
    labels = ["ham", "spam"]
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    metrics = {
        "dataset_records_after_cleaning": int(len(dataset)),
        "ham_records": int((y == "ham").sum()),
        "spam_records": int((y == "spam").sum()),
        "train_records": int(len(X_train)),
        "test_records": int(len(X_test)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "algorithm": "Multinomial Naive Bayes",
        "feature_extraction": "TF-IDF word unigrams and bigrams",
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_spam": float(precision_score(y_test, predictions, pos_label="spam", zero_division=0)),
        "recall_spam": float(recall_score(y_test, predictions, pos_label="spam", zero_division=0)),
        "f1_spam": float(f1_score(y_test, predictions, pos_label="spam", zero_division=0)),
        "precision_weighted": float(precision_score(y_test, predictions, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, predictions, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        "confusion_matrix_labels": labels,
        "confusion_matrix": matrix.tolist(),
    }

    # 7. Save the complete fitted pipeline (vectorizer + classifier).
    joblib.dump(model, MODEL_PATH)

    # 8. Save metrics and the detailed class-wise report.
    METRICS_PATH.write_text(json.dumps(metrics, indent=4), encoding="utf-8")
    REPORT_PATH.write_text(
        classification_report(y_test, predictions, labels=labels, zero_division=0),
        encoding="utf-8",
    )

    # 9. Create a readable confusion-matrix image in a light theme.
    sns.set_theme(style="white", font_scale=1.1)
    plt.figure(figsize=(6, 5), facecolor="white")
    axis = sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"],
        linewidths=1, linecolor="white",
    )
    axis.set_title("SMS Spam Detection - Confusion Matrix", pad=14)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("Actual label")
    plt.tight_layout()
    plt.savefig(MATRIX_PATH, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()

    print("\nTRAINING AND EVALUATION COMPLETED")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved confusion matrix: {MATRIX_PATH}")

if __name__ == "__main__":
    main()
