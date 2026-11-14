# SMS Shield — Text Classification using Machine Learning

## Project details
- **Domain:** Text Classification
- **Case study:** SMS Spam Detection
- **Algorithm:** Multinomial Naive Bayes
- **Feature extraction:** TF-IDF (unigrams + bigrams)
- **Dataset:** UCI SMS Spam Collection (CC BY 4.0)
- **Interface:** Streamlit, fully light themed

## What the project does
The application classifies an SMS message as `spam` or `ham` (legitimate). It downloads the authentic UCI dataset, validates and cleans it, performs a stratified 80:20 split, trains a TF-IDF + Multinomial Naive Bayes pipeline, calculates the required classification metrics and provides a live interface.

## Setup on Windows 11 / VS Code

Open this folder in VS Code. Open **Terminal → New Terminal** and run each command separately:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
python generate_report.py
```

The first training run downloads the original dataset. Internet access is required for this one-time download. If PowerShell blocks activation, use:

```powershell
.venv\Scripts\activate.bat
```

Then launch the app:

```powershell
streamlit run app.py
```

A browser tab should open. If it does not, copy the `Local URL` shown in the terminal (usually `http://localhost:8501`) into your browser.

## Files created after training
- `data/SMSSpamCollection.tsv` — downloaded original dataset
- `data/sms_spam_clean.csv` — cleaned data used for training/testing
- `models/sms_spam_pipeline.joblib` — fitted TF-IDF + Naive Bayes pipeline
- `outputs/metrics.json` — measured metrics and split details
- `outputs/classification_report.txt` — per-class report
- `outputs/confusion_matrix.png` — confusion matrix figure
- `PROJECT_REPORT_FINAL.pdf` — report populated with actual measured metrics (add your own live demo screenshots)

## Reproducibility and quality checks
- Fixed `random_state=42`
- Stratified 80:20 train-test split
- TF-IDF is fitted only on training data through a scikit-learn Pipeline
- Duplicate and empty messages are removed before splitting
- The test set is not used for fitting the model
- Model and metrics are saved locally for review
- Both classes are checked before training

## Dataset source and attribution
Almeida, T. & Hidalgo, J. (2011). *SMS Spam Collection* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84

License: Creative Commons Attribution 4.0 International (CC BY 4.0).

## Important
This is an educational model, not a production security filter. A high confidence score does not guarantee a message is safe. The dataset is English-language and reflects older SMS patterns.

## Code explanation
Read `LINE_BY_LINE_CODE_GUIDE.md`. The Python files also contain section headings and comments. The Jupyter notebook contains the same workflow in sequential cells.
