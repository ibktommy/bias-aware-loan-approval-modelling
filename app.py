import joblib
import numpy as np
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Bias-Aware Loan Approval Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚖️ Bias-Aware Loan Approval & Fairness Audit")
st.markdown(
    """
Evaluate credit risk and compare loan approval outcomes across **3 Mitigation Phases**:
- **Phase 1:** Baseline (Unmitigated)
- **Phase 2:** SMOTE Oversampled
- **Phase 3:** Reweighted + Threshold Tuned
"""
)


# -----------------------------------------------------------------------------
# ARTIFACT LOADING FROM HUGGING FACE HUB
# -----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
  repo_id = "atomdev-ibktommy/credit-bias-audit-models"
  artifact_files = {
      "scaler": "scaler.joblib",
      "rf_p1": "rf_phase1.joblib",
      "rf_p2": "rf_phase2.joblib",
      "rf_p3": "rf_phase3.joblib",
      "xgb_p1": "xgb_phase1.joblib",
      "xgb_p2": "xgb_phase2.joblib",
      "xgb_p3": "xgb_phase3.joblib",
  }

  loaded = {}
  for key, filename in artifact_files.items():
    try:
      file_path = hf_hub_download(repo_id=repo_id, filename=filename)
      loaded[key] = joblib.load(file_path)
    except Exception as e:
      st.sidebar.error(f"Failed to load {filename}: {e}")
  return loaded


artifacts = load_artifacts()

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Configuration & Model Selection")

selected_arch = st.sidebar.radio(
    "Select Model Architecture:",
    options=["Random Forest", "XGBoost"],
    index=0,
    help="Switch between the top candidate architectures to inspect their decisions.",
)

st.sidebar.markdown("---")
st.sidebar.header("👤 Applicant Data Entry")

# Form inputs for loan applicant features
income = st.sidebar.number_input(
    "Income ($)", min_value=10000, max_value=10000000, value=1300000, step=50000
)
age = st.sidebar.slider("Age", min_value=18, max_value=80, value=32)
experience = st.sidebar.slider(
    "Years of Work Experience", min_value=0, max_value=50, value=5
)
current_job_yrs = st.sidebar.slider(
    "Years at Current Job", min_value=0, max_value=30, value=3
)
current_house_yrs = st.sidebar.slider(
    "Years at Current Residence", min_value=0, max_value=30, value=10
)

married = st.sidebar.selectbox("Marital Status", ["single", "married"])
house_ownership = st.sidebar.selectbox(
    "House Ownership", ["rented", "owned", "norent_noown"]
)
car_ownership = st.sidebar.selectbox("Car Ownership", ["no", "yes"])

city_freq = st.sidebar.number_input(
    "City Frequency Count", min_value=1, max_value=5000, value=1250
)
state_freq = st.sidebar.number_input(
    "State Frequency Count", min_value=1, max_value=50000, value=15000
)

# Build raw input dictionary
raw_input = {
    "Income": income,
    "Age": age,
    "Experience": experience,
    "CURRENT_JOB_YRS": current_job_yrs,
    "CURRENT_HOUSE_YRS": current_house_yrs,
    "CITY_Freq": city_freq,
    "STATE_Freq": state_freq,
    "Married_Single_single": 1 if married == "single" else 0,
    "House_Ownership_owned": 1 if house_ownership == "owned" else 0,
    "House_Ownership_rented": 1 if house_ownership == "rented" else 0,
    "Car_Ownership_yes": 1 if car_ownership == "yes" else 0,
}

input_features = pd.DataFrame([raw_input])

# -----------------------------------------------------------------------------
# FEATURE ALIGNMENT & MODEL INFERENCE ENGINE
# -----------------------------------------------------------------------------
if artifacts and "scaler" in artifacts:
  scaler = artifacts["scaler"]

  # 1. Force input_features to strictly match the scaler's expected feature set & order
  if hasattr(scaler, "feature_names_in_"):
    expected_features = scaler.feature_names_in_
    input_features = input_features.reindex(
        columns=expected_features, fill_value=0.0
    )

  # 2. Scale features and reconstruct a named DataFrame to preserve feature metadata
  scaled_array = scaler.transform(input_features)
  scaled_inputs = pd.DataFrame(
      scaled_array, columns=input_features.columns, index=input_features.index
  )

  # Map keys based on user sidebar selection
  p1_key = "rf_p1" if selected_arch == "Random Forest" else "xgb_p1"
  p2_key = "rf_p2" if selected_arch == "Random Forest" else "xgb_p2"
  p3_key = "rf_p3" if selected_arch == "Random Forest" else "xgb_p3"

  # Optimal decision thresholds derived during Phase 3 fairness optimization
  p3_threshold = 0.51 if selected_arch == "Random Forest" else 0.61

  # 3. Compute predicted approval probabilities across phases
  prob_p1 = float(artifacts[p1_key].predict_proba(scaled_inputs)[:, 1][0])
  prob_p2 = float(artifacts[p2_key].predict_proba(scaled_inputs)[:, 1][0])
  prob_p3 = float(artifacts[p3_key].predict_proba(scaled_inputs)[:, 1][0])

  dec_p1 = prob_p1 >= 0.50
  dec_p2 = prob_p2 >= 0.50
  dec_p3 = prob_p3 >= p3_threshold

  # -----------------------------------------------------------------------------
  # DASHBOARD DISPLAY & METRIC COMPARISON
  # -----------------------------------------------------------------------------
  st.subheader(f"📊 Loan Approval Results — {selected_arch}")

  col1, col2, col3 = st.columns(3)

  def format_decision(is_approved):
    return "✅ Approved" if is_approved else "❌ Rejected"

  with col1:
    st.markdown("### Phase 1: Unmitigated")
    st.metric("Approval Probability", f"{prob_p1 * 100:.2f}%")
    st.subheader(format_decision(dec_p1))
    st.caption("Standard baseline model trained on unweighted raw data.")

  with col2:
    st.markdown("### Phase 2: SMOTE")
    st.metric(
        "Approval Probability",
        f"{prob_p2 * 100:.2f}%",
        delta=f"{(prob_p2 - prob_p1) * 100:+.2f}% vs P1",
    )
    st.subheader(format_decision(dec_p2))
    st.caption("Rebalanced using Synthetic Minority Over-sampling Technique.")

  with col3:
    st.markdown("### Phase 3: Reweighted + Tuned")
    st.metric(
        "Approval Probability",
        f"{prob_p3 * 100:.2f}%",
        delta=f"{(prob_p3 - prob_p1) * 100:+.2f}% vs P1",
    )
    st.subheader(format_decision(dec_p3))
    st.caption(
        f"Fairness reweighted & decision threshold calibrated at"
        f" {p3_threshold:.2f}."
    )

  st.divider()

  # Detailed feature inspection tab
  with st.expander("🔍 Inspect Processed Input Vectors"):
    st.markdown("**Original Raw Inputs:**")
    st.dataframe(pd.DataFrame([raw_input]))
    st.markdown("**Scaled Feature Matrix Passed to Classifier:**")
    st.dataframe(scaled_inputs)

else:
  st.error(
      "Model artifacts could not be loaded from Hugging Face Hub. Please check"
      " repository access and network connection."
  )