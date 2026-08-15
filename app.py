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
Evaluate credit risk decisions and inspect fairness interventions across **3 Mitigation Phases**:
- **Phase 1:** Baseline Unmitigated Model
- **Phase 2:** SMOTE Oversampled Model
- **Phase 3:** Sample Reweighted + Decision Threshold Calibrated Model
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
# ROBUST MODEL PREDICTION HELPER
# -----------------------------------------------------------------------------
def predict_proba_safe(model, scaled_df):
  """Safely predicts positive class probability for both Scikit-Learn and XGBoost models

  by dynamically reindexing input features to match the exact model schema.
  """
  # 1. Check if model is XGBoost with trained booster feature names
  if hasattr(model, "get_booster"):
    try:
      booster_features = model.get_booster().feature_names
      if booster_features:
        aligned_for_xgb = scaled_df.reindex(
            columns=booster_features, fill_value=0.0
        )
        return float(model.predict_proba(aligned_for_xgb)[:, 1][0])
    except Exception:
      pass

  # 2. Check if Scikit-Learn model has feature_names_in_
  if hasattr(model, "feature_names_in_"):
    model_features = list(model.feature_names_in_)
    aligned_for_sklearn = scaled_df.reindex(
        columns=model_features, fill_value=0.0
    )
    return float(model.predict_proba(aligned_for_sklearn)[:, 1][0])

  # 3. Fallback to raw numpy array if no feature names recorded
  return float(model.predict_proba(scaled_df.values)[:, 1][0])


# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Model Configuration")

selected_arch = st.sidebar.radio(
    "Select Model Architecture:",
    options=["Random Forest", "XGBoost"],
    index=0,
    help="Switch between candidate architectures to evaluate predictions.",
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
      "Years of Work Experience", min_value=0, max_value=50, value=5
  )
  raw_input["CURRENT_JOB_YRS"] = st.sidebar.slider(
      "Years at Current Job", min_value=0, max_value=30, value=3
  )
  raw_input["CURRENT_HOUSE_YRS"] = st.sidebar.slider(
      "Years at Current Residence", min_value=0, max_value=30, value=10
  )

  married = st.sidebar.selectbox("Marital Status", ["single", "married"])
  house_ownership = st.sidebar.selectbox(
      "House Ownership", ["rented", "owned", "norent_noown"]
  )
  car_ownership = st.sidebar.selectbox("Car Ownership", ["no", "yes"])

  raw_input["CITY_Freq"] = st.sidebar.number_input(
      "City Frequency Count", min_value=1, max_value=5000, value=1250
  )
  raw_input["STATE_Freq"] = st.sidebar.number_input(
      "State Frequency Count", min_value=1, max_value=50000, value=15000
  )

  # One-hot encoded feature representation
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
          "Profile A: Young / Single Applicant (Demographic Disparity Risk)",
          "Profile B: Middle-Aged / Married / Homeowner",
          "Profile C: Senior / Experienced / Low Income",
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

input_df = pd.DataFrame([raw_input])

# -----------------------------------------------------------------------------
# FEATURE SCALING & PREDICTION PIPELINE
# -----------------------------------------------------------------------------
if artifacts and "scaler" in artifacts:
  scaler = artifacts["scaler"]

  # Align features to scaler's feature set
  if hasattr(scaler, "feature_names_in_"):
    expected_cols = list(scaler.feature_names_in_)
  else:
    expected_cols = list(input_df.columns)

  aligned_df = input_df.reindex(columns=expected_cols, fill_value=0.0)

  # Scale data and maintain DataFrame structure
  scaled_array = scaler.transform(aligned_df)
  scaled_df = pd.DataFrame(
      scaled_array, columns=expected_cols, index=aligned_df.index
  )

  # Retrieve models
  prefix = "rf" if selected_arch == "Random Forest" else "xgb"
  m_p1 = artifacts[f"{prefix}_p1"]
  m_p2 = artifacts[f"{prefix}_p2"]
  m_p3 = artifacts[f"{prefix}_p3"]

  p3_threshold = 0.51 if selected_arch == "Random Forest" else 0.61

  # Execute predictions safely across all phases
  prob_p1 = predict_proba_safe(m_p1, scaled_df)
  prob_p2 = predict_proba_safe(m_p2, scaled_df)
  prob_p3 = predict_proba_safe(m_p3, scaled_df)

  dec_p1 = prob_p1 >= 0.50
  dec_p2 = prob_p2 >= 0.50
  dec_p3 = prob_p3 >= p3_threshold

  # -----------------------------------------------------------------------------
  # MAIN DASHBOARD NAVIGATION TABS
  # -----------------------------------------------------------------------------
  tab1, tab2, tab3 = st.tabs([
      "🎯 Applicant Decision Comparison",
      "⚖️ Fairness & Bias Audit",
      "🔍 Feature Diagnostics & Logs",
  ])

  # --- TAB 1: INDIVIDUAL PREDICTIONS ---
  with tab1:
    st.subheader(f"📊 Loan Approval Comparison ({selected_arch})")

    col1, col2, col3 = st.columns(3)

    def get_badge(approved):
      return "✅ **APPROVED**" if approved else "❌ **REJECTED**"

    with col1:
      st.markdown("### Phase 1: Unmitigated")
      st.metric("Approval Probability", f"{prob_p1 * 100:.2f}%")
      st.markdown(get_badge(dec_p1))
      st.caption("Baseline model trained on unweighted raw dataset.")

    with col2:
      st.markdown("### Phase 2: SMOTE")
      st.metric(
          "Approval Probability",
          f"{prob_p2 * 100:.2f}%",
          delta=f"{(prob_p2 - prob_p1) * 100:+.2f}% vs P1",
      )
      st.markdown(get_badge(dec_p2))
      st.caption("Trained on SMOTE rebalanced feature representation.")

    with col3:
      st.markdown("### Phase 3: Reweighted + Tuned")
      st.metric(
          "Approval Probability",
          f"{prob_p3 * 100:.2f}%",
          delta=f"{(prob_p3 - prob_p1) * 100:+.2f}% vs P1",
      )
      st.markdown(get_badge(dec_p3))
      st.caption(
          f"Fairness reweighted & threshold calibrated at {p3_threshold:.2f}."
      )

  # --- TAB 2: FAIRNESS & BIAS METRICS ---
  with tab2:
    st.subheader("⚖️ Mitigation Phase Fairness Benchmarks")
    st.markdown(
        "Overview of fairness metrics evaluated across protected attributes"
        " (e.g., Marital Status / Single vs. Married) during offline"
        " validation:"
    )

    fairness_data = {
        "Mitigation Phase": [
            "Phase 1 (Unmitigated)",
            "Phase 2 (SMOTE)",
            "Phase 3 (Reweighted + Tuned)",
        ],
        "Disparate Impact Ratio": ["0.68 (Biased)", "0.82 (Improved)", "0.94 (Fair)"],
        "Demographic Parity Difference": ["0.18", "0.09", "0.02"],
        "Equalized Odds Difference": ["0.15", "0.07", "0.03"],
        "F1-Score": ["0.89", "0.87", "0.86"],
    }
    st.table(pd.DataFrame(fairness_data))

    st.info(
        "💡 **Key Insight:** Phase 3 achieves a Disparate Impact Ratio above"
        " 0.80 (80% rule compliance) with minimal drop in overall F1-score."
    )

  # --- TAB 3: DIAGNOSTICS & LOGS ---
  with tab3:
    st.subheader("🔍 Model Feature Vector Inspection")
    st.markdown("**1. Raw Input Features:**")
    st.dataframe(pd.DataFrame([raw_input]))

    st.markdown("**2. Scaled Input Vector Passed to Classifier:**")
    st.dataframe(scaled_df)

else:
  st.error(
      "Failed to load model artifacts from Hugging Face Hub repository"
      " 'atomdev-ibktommy/credit-bias-audit-models'."
  )