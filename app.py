
"""
LIGHT-THEME STREAMLIT APPLICATION
Run: streamlit run app.py
The app can also train the model automatically if it is missing.
"""

from pathlib import Path
import subprocess
import sys
import json
import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "sms_spam_pipeline.joblib"
METRICS_PATH = ROOT / "outputs" / "metrics.json"

st.set_page_config(
    page_title="SMS Shield | Spam Detection",
    page_icon="🛡️",
    layout="wide",
)

# All styling is intentionally light; no dark-theme components are used.
st.markdown("""
<style>
.stApp { background: #FFFFFF; color: #172033; }
.block-container { max-width: 1120px; padding-top: 2rem; }
.hero { background: #F2F5FF; border: 1px solid #DCE5FF; padding: 1.5rem;
        border-radius: 18px; margin-bottom: 1.2rem; }
.hero h1 { color: #172B62; margin: 0; }
.hero p { color: #52617D; margin-bottom: 0; }
.result { padding: 1rem 1.2rem; border-radius: 14px; margin-top: 1rem;
          border: 1px solid #DDE3EE; background: #F8FAFD; }
.spam { background: #FFF1F0; border-color: #F4C7C3; }
.ham { background: #ECFDF3; border-color: #B7E4C7; }
div[data-testid="stMetric"] { background: #F6F8FC; padding: 1rem;
    border-radius: 12px; border: 1px solid #E5EAF2; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the saved scikit-learn pipeline; train it if it does not exist."""
    if not MODEL_PATH.exists():
        with st.spinner("First-time setup: downloading data and training the model..."):
            result = subprocess.run(
                [sys.executable, str(ROOT / "train_model.py")],
                cwd=ROOT, capture_output=True, text=True,
            )
            if result.returncode != 0:
                st.error("Model setup failed. Check your internet connection and terminal logs.")
                st.code(result.stderr or result.stdout)
                st.stop()

    return joblib.load(MODEL_PATH)


st.markdown("""
<div class="hero">
  <h1>SMS Shield</h1>
  <p>Machine-learning powered SMS spam detection using TF-IDF and Multinomial Naive Bayes.</p>
</div>
""", unsafe_allow_html=True)

model = load_model()

left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.subheader("Test a message")

    message = st.text_area(
        "Enter or paste an SMS message",
        placeholder="Example: Congratulations! You have won a free prize. Call now...",
        height=170,
        help="Enter one message in English. The model was trained on the UCI SMS dataset.",
    )

    predict_clicked = st.button(
        "Classify message",
        type="primary",
        use_container_width=True
    )

    if predict_clicked:
        if not message.strip():
            st.warning("Please enter a message before clicking Classify.")
        else:
            prediction = model.predict([message])[0]
            probabilities = model.predict_proba([message])[0]
            classes = list(model.classes_)

            confidence = float(
                probabilities[classes.index(prediction)]
            ) * 100

            if prediction == "spam":
                st.markdown(
                    f'<div class="result spam"><h3>Spam message detected</h3>'
                    f'<p>Predicted class: <b>SPAM</b> · Model confidence: <b>{confidence:.1f}%</b></p>'
                    f'<p>Be cautious with links, prize claims, and requests for personal information.</p></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="result ham"><h3>Not Spam</h3>'
                    f'<p>Predicted class: <b>NOT SPAM</b> · Model confidence: <b>{confidence:.1f}%</b></p>'
                    f'<p>This message resembles legitimate messages in the training dataset.</p></div>',
                    unsafe_allow_html=True,
                )

            st.caption(
                "Confidence is the model's estimated class probability, not a guarantee."
            )


with right:
    st.subheader("How the system works")

    st.markdown("""
    1. **Input:** You enter an SMS message.
    2. **TF-IDF:** Text is converted into numerical word features.
    3. **Classification:** The trained system analyses the message and identifies its category.
    4. **Output:** The app displays Spam or Not Spam with a confidence estimate.
    """)

    st.info(
        "This is an educational classifier. Do not use it as the sole basis for security decisions."
    )


st.divider()
st.subheader("Model evaluation")

if METRICS_PATH.exists():
    metrics = json.loads(
        METRICS_PATH.read_text(encoding="utf-8")
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Accuracy",
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    c2.metric(
        "Precision (Spam)",
        f"{metrics['precision_spam'] * 100:.2f}%"
    )

    c3.metric(
        "Recall (Spam)",
        f"{metrics['recall_spam'] * 100:.2f}%"
    )

    c4.metric(
        "F1-score (Spam)",
        f"{metrics['f1_spam'] * 100:.2f}%"
    )

    st.caption(
        f"Clean records: {metrics['dataset_records_after_cleaning']:,} · "
        f"Training: {metrics['train_records']:,} · "
        f"Testing: {metrics['test_records']:,} · "
        f"Split: 80:20 · Random state: 42"
    )

    matrix_path = ROOT / "outputs" / "confusion_matrix.png"

    if matrix_path.exists():
        st.image(
            str(matrix_path),
            caption="Confusion matrix from the held-out test set",
            width=520
        )

else:
    st.warning(
        "Evaluation metrics will appear after the first successful model training."
    )


st.caption(
    "Dataset: UCI SMS Spam Collection · Educational project · Light theme"
)
