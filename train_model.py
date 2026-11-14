
"""
SMS SPAM DETECTION - NAIVE BAYES MODEL

Feature extraction: TF-IDF
Algorithm: Multinomial Naive Bayes
Train-test split: 80:20
"""

from pathlib import Path
import json
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
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

LABELS = ["ham", "spam"]


def main():

    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load the cleaned SMS dataset.
    dataset = load_and_clean_dataset()

    X = dataset["message"]
    y = dataset["label"]

    # Split the dataset into training and testing data.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Create the text classification pipeline.
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.98,
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            MultinomialNB(alpha=0.5)
        )
    ])

    # Train the model.
    model.fit(X_train, y_train)

    # Predict the test messages.
    predictions = model.predict(X_test)

    # Calculate the confusion matrix.
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=LABELS
    )

    # Calculate evaluation metrics.
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

        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),

        "precision_spam": float(
            precision_score(
                y_test, predictions,
                pos_label="spam",
                zero_division=0
            )
        ),

        "recall_spam": float(
            recall_score(
                y_test, predictions,
                pos_label="spam",
                zero_division=0
            )
        ),

        "f1_spam": float(
            f1_score(
                y_test, predictions,
                pos_label="spam",
                zero_division=0
            )
        ),

        "precision_weighted": float(
            precision_score(
                y_test, predictions,
                average="weighted",
                zero_division=0
            )
        ),

        "recall_weighted": float(
            recall_score(
                y_test, predictions,
                average="weighted",
                zero_division=0
            )
        ),

        "f1_weighted": float(
            f1_score(
                y_test, predictions,
                average="weighted",
                zero_division=0
            )
        ),

        "confusion_matrix_labels": [
            "Not Spam",
            "Spam"
        ],

        "confusion_matrix": matrix.tolist()
    }

    # Save the trained model.
    joblib.dump(model, MODEL_PATH)

    # Save the evaluation metrics.
    METRICS_PATH.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8"
    )

    # Save the classification report.
    report = classification_report(
        y_test,
        predictions,
        labels=LABELS,
        target_names=["Not Spam", "Spam"],
        zero_division=0
    )

    REPORT_PATH.write_text(
        report,
        encoding="utf-8"
    )

    # Create a light-theme confusion matrix.
    sns.set_theme(style="white", font_scale=1.1)

    plt.figure(figsize=(6, 5), facecolor="white")

    axis = sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Not Spam", "Spam"],
        yticklabels=["Not Spam", "Spam"],
        linewidths=1,
        linecolor="white"
    )

    axis.set_title(
        "SMS Spam Detection - Confusion Matrix",
        pad=14
    )

    axis.set_xlabel("Predicted category")
    axis.set_ylabel("Actual category")

    plt.tight_layout()

    plt.savefig(
        MATRIX_PATH,
        dpi=180,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close()

    # Display the results.
    print("\nTRAINING AND EVALUATION COMPLETED")

    print(json.dumps(metrics, indent=4))

    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved confusion matrix: {MATRIX_PATH}")


if __name__ == "__main__":
    main()
