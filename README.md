# NewsLens — Fake News Detection

A supervised NLP application that predicts whether a news article resembles the Fake or Real examples in a labelled historical dataset. It includes a Streamlit interface, TF-IDF features, model comparison, and evaluation visualizations.

> Important: This is an educational text-classification model, not a fact-checking service. It does not verify claims against live sources. Predictions reflect patterns in the dataset and must not be treated as proof that an article is true or false.

## Dataset

This project uses the ISOT Fake and Real News Dataset:
- Fake.csv: articles labelled Fake.
- True.csv: articles labelled Real.
- Both files contain headlines and article text.
- The loader downloads the files on first use from a public GitHub mirror. Raw data is not committed to this repository.

Dataset background: the ISOT description says real articles were collected from Reuters and fake articles from sources identified as unreliable. The dataset is historical and predominantly covers political and world news.

## Workflow

1. Download labelled Fake and Real CSV files.
2. Combine headline and article body.
3. Remove missing, very short, and duplicate article text.
4. Extract TF-IDF word unigram and bigram features.
5. Train and compare Multinomial Naive Bayes and Logistic Regression.
6. Select the model with the higher macro F1-score.
7. Evaluate on a stratified held-out test set.
8. Save the model, metrics, report, and confusion matrix.

## Structure

- app.py — Streamlit interface
- src_data_loader.py — dataset download and cleaning
- train_model.py — model training and evaluation
- data/ — downloaded and cleaned data (not committed)
- models/ — saved trained model (not committed)
- outputs/ — metrics and visualizations (not committed)
- PROJECT_REPORT.md — case study report
- Fake_News_Classification.ipynb — Jupyter notebook

## Run locally (Windows PowerShell)

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    python src_data_loader.py
    python train_model.py
    streamlit run app.py

The first dataset download may take a few minutes. The app can trigger training automatically if the model is missing.

## Evaluation

The training script compares accuracy, macro precision, macro recall, and macro F1-score. It selects the model with the higher macro F1-score. Actual scores are generated when training runs; no performance figures are assumed in advance.

## Responsible use

- A high confidence score is not a probability that a claim is factually true.
- Dataset labels and source composition may introduce bias.
- The historical data may not represent current events or new misinformation techniques.
- Verify consequential claims with multiple reputable and independent sources.

## Dataset citation

ISOT Fake News Dataset, Information Security and Object Technology (ISOT) Research Group, University of Victoria:
https://onlineacademiccommunity.uvic.ca/isot/wp-content/uploads/sites/7295/2023/02/ISOT_Fake_News_Dataset_ReadMe.pdf

Public mirror used by the loader:
https://github.com/Reinforz/Fake-News-detection-with-ISOT-Dataset
