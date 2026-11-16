from pathlib import Path
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from src_data_loader import load_and_clean_dataset

ROOT = Path(__file__).resolve().parent
MODEL_DIR, OUTPUT_DIR = ROOT / "models", ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

def make_pipeline(classifier):
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode",
            ngram_range=(1, 2), min_df=2, max_df=0.95,
            max_features=100000, sublinear_tf=True)),
        ("classifier", classifier)
    ])

def main():
    data = load_and_clean_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        data["content"], data["label"], test_size=0.20,
        random_state=42, stratify=data["label"]
    )
    candidates = {
        "Multinomial Naive Bayes": make_pipeline(MultinomialNB(alpha=0.5)),
        "Logistic Regression": make_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    }
    scores, trained = {}, {}
    for name, model in candidates.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        scores[name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision_macro": float(precision_score(y_test, pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0))
        }
        trained[name] = model
        print(scores[name])
    best_name = max(scores, key=lambda key: scores[key]["f1_macro"])
    best_model = trained[best_name]
    predictions = best_model.predict(X_test)
    labels = ["Fake", "Real"]
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    joblib.dump(best_model, MODEL_DIR / "fake_news_pipeline.joblib")
    report = classification_report(y_test, predictions, labels=labels, target_names=labels, zero_division=0)
    (OUTPUT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    plt.figure(figsize=(7, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()
    metrics = {
        "dataset": "ISOT Fake and Real News Dataset",
        "dataset_records_after_cleaning": int(len(data)),
        "fake_records": int((data["label"] == "Fake").sum()),
        "real_records": int((data["label"] == "Real").sum()),
        "train_records": int(len(X_train)), "test_records": int(len(X_test)),
        "test_size": 0.20, "random_state": 42,
        "feature_extraction": "TF-IDF word unigrams and bigrams",
        "models": scores, "best_model": best_name,
        **scores[best_name],
        "confusion_matrix_labels": labels, "confusion_matrix": matrix.tolist(),
        "note": "Predictions reflect patterns learned from this historical dataset; they are not independent fact-checks."
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=4), encoding="utf-8")
    print("\nTRAINING AND EVALUATION COMPLETED")
    print(json.dumps(metrics, indent=4))
    print(f"Saved model: {MODEL_DIR / 'fake_news_pipeline.joblib'}")
    print(f"Saved metrics: {OUTPUT_DIR / 'metrics.json'}")
    print(f"Saved confusion matrix: {OUTPUT_DIR / 'confusion_matrix.png'}")

if __name__ == "__main__":
    main()
