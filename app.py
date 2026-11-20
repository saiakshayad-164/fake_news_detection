from pathlib import Path
import json
import subprocess
import sys
import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "fake_news_pipeline.joblib"
METRICS_PATH = ROOT / "outputs" / "metrics.json"
REPORT_PATH = ROOT / "outputs" / "classification_report.txt"
MATRIX_PATH = ROOT / "outputs" / "confusion_matrix.png"

st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="wide")
st.markdown("""
<style>
.stApp {background:#f7f9fc;color:#172033}
.block-container {max-width:1180px;padding-top:2rem;padding-bottom:3rem}
.hero {background:linear-gradient(120deg,#fff,#edf4ff);border:1px solid #dce6f5;padding:2rem;border-radius:18px;margin-bottom:1.3rem}
.hero h1 {color:#14213d;margin:0;font-size:2.25rem}
.hero p {color:#53627a;margin:.6rem 0 0;font-size:1.05rem}
.panel {background:#fff;border:1px solid #e2e8f0;padding:1.2rem;border-radius:14px}
.result-real {background:#eaf8ef;border:1px solid #a8dfba;color:#176638;padding:1.2rem;border-radius:12px}
.result-fake {background:#fff0f0;border:1px solid #f2b8b8;color:#a12626;padding:1.2rem;border-radius:12px}
.small-note {color:#64748b;font-size:.9rem}
div[data-testid="stMetric"] {background:#fff;border:1px solid #e2e8f0;padding:14px;border-radius:12px}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        with st.spinner("Downloading the news dataset and training models for the first run..."):
            result = subprocess.run([sys.executable, str(ROOT / "train_model.py")],
                cwd=ROOT, capture_output=True, text=True)
            if result.returncode != 0:
                st.error("Model setup failed. Details:")
                st.code(result.stderr or result.stdout)
                st.stop()
    return joblib.load(MODEL_PATH)

def read_metrics():
    return json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else None

st.markdown('<div class="hero"><h1>NewsLens — Fake News Detection</h1><p>Analyse a news article using natural language processing and machine learning.</p></div>', unsafe_allow_html=True)
with st.sidebar:
    st.header("About this project")
    st.write("An educational classifier trained on labelled historical news articles.")
    st.info("A prediction is not proof that a story is true or false. Verify important claims with multiple reliable sources.")
    st.caption("Dataset: ISOT Fake and Real News")

left, right = st.columns([1.45, 1], gap="large")
with left:
    st.subheader("Analyse a news article")
    title = st.text_input("News headline (optional)", placeholder="Enter the article headline")
    article = st.text_area("News article text", height=270, placeholder="Paste the full news article here...")
    classify = st.button("Classify News", type="primary", use_container_width=True)
with right:
    st.subheader("How it works")
    st.markdown("""
    <div class="panel">
    <b>1. Text input</b><br><span class="small-note">Provide a headline and article body.</span><br><br>
    <b>2. TF-IDF features</b><br><span class="small-note">Convert words and phrases into numerical features.</span><br><br>
    <b>3. Classification</b><br><span class="small-note">The selected model predicts Fake or Real based on learned patterns.</span>
    </div>""", unsafe_allow_html=True)

if classify:
    combined = (title.strip() + " " + article.strip()).strip()
    if len(combined) < 30:
        st.warning("Please enter a headline or article with at least 30 characters.")
    else:
        model = load_model()
        prediction = model.predict([combined])[0]
        probabilities = model.predict_proba([combined])[0]
        confidence = float(probabilities[list(model.classes_).index(prediction)])
        if prediction == "Real":
            st.markdown(f'<div class="result-real"><h2>Prediction: REAL NEWS</h2><p>Model confidence: {confidence:.1%}</p><p>Based on patterns learned from the training dataset.</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-fake"><h2>Prediction: FAKE NEWS</h2><p>Model confidence: {confidence:.1%}</p><p>Based on patterns learned from the training dataset.</p></div>', unsafe_allow_html=True)
        st.progress(confidence, text="Model confidence (not factual certainty)")
        st.caption("This is a machine-learning prediction, not independent verification of the article.")

st.divider()
st.header("Model evaluation")
metrics = read_metrics()
if metrics:
    st.caption(f"Best model: {metrics['best_model']} | Cleaned records: {metrics['dataset_records_after_cleaning']:,}")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    c2.metric("Macro precision", f"{metrics['precision_macro']:.2%}")
    c3.metric("Macro recall", f"{metrics['recall_macro']:.2%}")
    c4.metric("Macro F1-score", f"{metrics['f1_macro']:.2%}")
    st.subheader("Algorithm comparison")
    rows = [{"Model":name, **{key.replace("_"," ").title():f"{value:.2%}" for key,value in values.items()}} for name,values in metrics["models"].items()]
    st.dataframe(rows, use_container_width=True, hide_index=True)
    if MATRIX_PATH.exists():
        st.subheader("Confusion matrix")
        st.image(str(MATRIX_PATH), use_container_width=True)
    if REPORT_PATH.exists():
        with st.expander("View classification report"):
            st.text(REPORT_PATH.read_text(encoding="utf-8"))
else:
    st.info("Evaluation results appear after the first successful training run.")
st.divider()
st.markdown('<p class="small-note">Educational demonstration only. Dataset labels reflect the dataset construction and are not independent fact-checks of each article.</p>', unsafe_allow_html=True)
