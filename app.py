import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
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
    Evaluate credit risk decisions across **3 Mitigation Phases**:
    - **Phase 1:** Baseline Unmitigated
    - **Phase 2:** SMOTE Oversampled
    - **Phase 3:** Reweighted + Decision Threshold Tuned
    """
)

# -----------------------------------------------------------------------------
# ARTIFACT LOADING FROM HUGGING FACE HUB
# -----------------------------------------------------------------------------
REPO_ID = "atomdev-ibktommy/credit-bias-audit-models"


@st.cache_resource
def load_artifacts():
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
      file_path = hf_hub_download(repo_id=REPO_ID, filename=filename)
      loaded[key] = joblib.load(file_path)
    except Exception as e:
      st.sidebar.error(f"Error loading {filename}: {e}")
  return loaded


artifacts = load_artifacts()

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS & INPUT MODES
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Model Configuration")

selected_arch = st.sidebar.radio(
    "Select Model Architecture:",
    options=["Random Forest", "XGBoost"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.header("📥 Applicant Input Mode")

input_mode = st.sidebar.radio(
    "Choose Input Method:",
    options=["Manual Form Entry", "Sample Test Preset"],
)

raw_input = {}

if input_mode == "Manual Form Entry":
  st.sidebar.subheader("👤 Enter Applicant Details")
  raw_input["Income"] = st.sidebar.number_input(
      "Income ($)",
      min_value=10000,
      max_value=10000000,
      value=1300000,
      step=50000,
  )
  raw_input["Age"] = st.sidebar.slider(
      "Age", min_value=18, max_value=80, value=32
  )
  raw_input["Experience"] = st.sidebar.slider(
      "Years of Experience", min_value=0, max_value=50, value=5
  )
  raw_input["CURRENT_JOB_YRS"] = st.sidebar.slider(
      "Years at Current Job", min_value=0, max_value=30, value=3
  )
  raw_input["CURRENT_HOUSE_YRS"] = st.sidebar.slider(
      "Years at Current House", min_value=0, max_value=30, value=10
  )

  married = st.sidebar.selectbox("Marital Status", ["single", "married"])
  house_ownership = st.sidebar.selectbox(
      "House Ownership", ["rented", "owned", "norent_noown"]
  )
  car_ownership = st.sidebar.selectbox("Car Ownership", ["no", "yes"])

  raw_input["CITY_Freq"] = st.sidebar.number_input(
      "City Frequency", min_value=1, max_value=5000, value=1250
  )
  raw_input["STATE_Freq"] = st.sidebar.number_input(
      "State Frequency", min_value=1, max_value=50000, value=15000
  )

  # Explicit One-Hot Encoding values
  raw_input["Married_Single_single"] = 1 if married == "single" else 0
  raw_input["Married_Single_married"] = 1 if married == "married" else 0
  raw_input["House_Ownership_owned"] = 1 if house_ownership == "owned" else 0
  raw_input["House_Ownership_rented"] = 1 if house_ownership == "rented" else 0
  raw_input["House_Ownership_norent_noown"] = (
      1 if house_ownership == "norent_noown" else 0
  )
  raw_input["Car_Ownership_yes"] = 1 if car_ownership == "yes" else 0
  raw_input["Car_Ownership_no"] = 1 if car_ownership == "no" else 0

else:
  st.sidebar.subheader("📋 Select Test Applicant Profile")
  preset = st.sidebar.selectbox(
      "Sample Profiles:",
      options=[
          "Profile A: High Income / Young / Single (Potential Bias Target)",
          "Profile B: Moderate Income / Experienced / Married",
          "Profile C: Low Income / High Experience / Rented",
      ],
  )

  if "Profile A" in preset:
    raw_input = {
        "Income": 850000,
        "Age": 24,
        "Experience": 2,
        "CURRENT_JOB_YRS": 2,
        "CURRENT_HOUSE_YRS": 3,
        "CITY_Freq": 800,
        "STATE_Freq": 12000,
        "Married_Single_single": 1,
        "Married_Single_married": 0,
        "House_Ownership_rented": 1,
        "House_Ownership_owned": 0,
        "House_Ownership_norent_noown": 0,
        "Car_Ownership_no": 1,
        "Car_Ownership_yes": 0,
    }
  elif "Profile B" in preset:
    raw_input = {
        "Income": 4500000,
        "Age": 42,
        "Experience": 18,
        "CURRENT_JOB_YRS": 10,
        "CURRENT_HOUSE_YRS": 12,
        "CITY_Freq": 2100,
        "STATE_Freq": 28000,
        "Married_Single_single": 0,
        "Married_Single_married": 1,
        "House_Ownership_owned": 1,
        "House_Ownership_rented": 0,
        "House_Ownership_norent_noown": 0,
        "Car_Ownership_yes": 1,
        "Car_Ownership_no": 0,
    }
  else:
    raw_input = {
        "Income": 1200000,
        "Age": 55,
        "Experience": 25,
        "CURRENT_JOB_YRS": 15,
        "CURRENT_HOUSE_YRS": 20,
        "CITY_Freq": 500,
        "STATE_Freq": 8000,
        "Married_Single_single": 1,
        "Married_Single_married": 0,
        "House_Ownership_rented": 1,
        "House_Ownership_owned": 0,
        "House_Ownership_norent_noown": 0,
        "Car_Ownership_yes": 0,
        "Car_Ownership_no": 1,
    }

# Convert user raw input into a single-row DataFrame
input_df = pd.DataFrame([raw_input])

# -----------------------------------------------------------------------------
# DYNAMIC FEATURE ALIGNMENT & SCALING ENGINE
# -----------------------------------------------------------------------------
if artifacts and "scaler" in artifacts:
  scaler = artifacts["scaler"]

  # Retrieve exact feature names and sequence expected by the fitted scaler
  if hasattr(scaler, "feature_names_in_"):
    expected_cols = list(scaler.feature_names_in_)
  else:
    expected_cols = list(input_df.columns)

  # Reindex DataFrame to guarantee exact column match and ordering
  aligned_df = input_df.reindex(columns=expected_cols, fill_value=0.0)

  # Scale data and wrap back into a DataFrame with feature names preserved
  scaled_array = scaler.transform(aligned_df)
  scaled_df = pd.DataFrame(
      scaled_array, columns=expected_cols, index=aligned_df.index
  )

  # Determine artifact keys based on selected architecture
  prefix = "rf" if selected_arch == "Random Forest" else "xgb"
  p1_key, p2_key, p3_key = f"{prefix}_p1", f"{prefix}_p2", f"{prefix}_p3"

  # Calibrated decision thresholds
  p3_threshold = 0.51 if selected_arch == "Random Forest" else 0.61

  # Model probability predictions
  prob_p1 = float(artifacts[p1_key].predict_proba(scaled_df)[:, 1][0])
  prob_p2 = float(artifacts[p2_key].predict_proba(scaled_df)[:, 1][0])
  prob_p3 = float(artifacts[p3_key].predict_proba(scaled_df)[:, 1][0])

  dec_p1 = prob_p1 >= 0.50
  dec_p2 = prob_p2 >= 0.50
  dec_p3 = prob_p3 >= p3_threshold

  # -----------------------------------------------------------------------------
  # DASHBOARD METRICS DISPLAY
  # -----------------------------------------------------------------------------
  st.subheader(f"📊 Loan Approval Comparison ({selected_arch})")

  col1, col2, col3 = st.columns(3)

  def get_status_badge(approved):
    return "✅ **Approved**" if approved else "❌ **Rejected**"

  with col1:
    st.markdown("### Phase 1: Unmitigated")
    st.metric("Approval Probability", f"{prob_p1 * 100:.2f}%")
    st.markdown(get_status_badge(dec_p1))
    st.caption("Standard baseline model trained on unweighted data.")

  with col2:
    st.markdown("### Phase 2: SMOTE Oversampled")
    st.metric(
        "Approval Probability",
        f"{prob_p2 * 100:.2f}%",
        delta=f"{(prob_p2 - prob_p1) * 100:+.2f}% vs P1",
    )
    st.markdown(get_status_badge(dec_p2))
    st.caption("Trained on SMOTE rebalanced feature set.")

  with col3:
    st.markdown("### Phase 3: Reweighted + Tuned")
    st.metric(
        "Approval Probability",
        f"{prob_p3 * 100:.2f}%",
        delta=f"{(prob_p3 - prob_p1) * 100:+.2f}% vs P1",
    )
    st.markdown(get_status_badge(dec_p3))
    st.caption(
        f"Sample reweighted and decision threshold calibrated at"
        f" {p3_threshold:.2f}."
    )

  st.divider()

  # Diagnostic Feature View
  with st.expander("🔍 Inspect Processed Input Vectors"):
    st.markdown("**Aligned Input DataFrame (Raw):**")
    st.dataframe(aligned_df)
    st.markdown("**Scaled Feature Matrix (Passed to Model):**")
    st.dataframe(scaled_df)

else:
  st.error("Failed to load model artifacts. Check Hugging Face repository.")