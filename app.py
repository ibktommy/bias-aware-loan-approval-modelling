import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Bias-Aware Credit Risk & Regulatory Parity Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .status-badge {
        background-color: #e6f4ea;
        color: #137333;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 8px;
    }
</style>
""",
    unsafe_allow_html=True,
)

REPO_ID = "atomdev-ibktommy/credit-bias-audit-models"


# -----------------------------------------------------------------------------
# 2. DYNAMIC ARTIFACT LOADER
# -----------------------------------------------------------------------------
@st.cache_resource
def load_hf_artifacts():
  """Downloads model artifacts, scaler, and test set directly from Hugging Face Hub."""
  artifact_files = {
      "scaler": "scaler.joblib",
      "test_df": "test_dataset.csv",
      "rf_p1": "rf_phase1.joblib",
      "xgb_p1": "xgb_phase1.joblib",
      "rf_p2": "rf_phase2.joblib",
      "xgb_p2": "xgb_phase2.joblib",
      "rf_p3": "rf_phase3.joblib",
      "xgb_p3": "xgb_phase3.joblib",
  }

  loaded_artifacts = {}
  source_info = ""

  try:
    for key, filename in artifact_files.items():
      downloaded_path = hf_hub_download(
          repo_id=REPO_ID, repo_type="model", filename=filename
      )
      if filename.endswith(".joblib"):
        loaded_artifacts[key] = joblib.load(downloaded_path)
      elif filename.endswith(".csv"):
        loaded_artifacts[key] = pd.read_csv(downloaded_path)

    source_info = f"Hugging Face Hub Live (`{REPO_ID}`)"

  except Exception as hf_err:
    try:
      local_dir = "deployment_artifacts"
      for key, filename in artifact_files.items():
        local_path = os.path.join(local_dir, filename)
        if filename.endswith(".joblib"):
          loaded_artifacts[key] = joblib.load(local_path)
        elif filename.endswith(".csv"):
          loaded_artifacts[key] = pd.read_csv(local_path)
      source_info = "Local Disk Storage (`deployment_artifacts/`)"
    except Exception as local_err:
      st.error(
          f"Failed to load artifacts from HF Hub ({hf_err}) and local"
          f" directory ({local_err})."
      )
      return None, "Error Loading Models"

  return loaded_artifacts, source_info


# Load all artifacts into memory
artifacts, status_msg = load_hf_artifacts()

# -----------------------------------------------------------------------------
# 3. GLOBAL SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.title("🛠️ Audit Engine Controls")
st.sidebar.markdown(
    f'<span class="status-badge">● {status_msg}</span>', unsafe_allow_html=True
)
st.sidebar.write("")

selected_arch = st.sidebar.radio(
    "Choose Classifier Architecture:",
    ["Random Forest", "XGBoost"],
    help="Select the underlying ensemble model to test across all 3 mitigation phases.",
)

input_mode = st.sidebar.selectbox(
    "Select Input Mode:",
    ["Test Set Sample Mode", "Manual Profile Input"],
    help="Switch between live sampling from the Hugging Face test set or manual feature entry.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Phase 3 Decision Thresholds ($t$)")
st.sidebar.info(
    "**Fairness-Tuned Cutoffs:**\n"
    "- Random Forest: **t = 0.51**\n"
    "- XGBoost: **t = 0.61**"
)

# -----------------------------------------------------------------------------
# 4. MAIN APPLICATION INTERFACE
# -----------------------------------------------------------------------------
st.title("⚖️ Bias-Aware Credit Risk & Regulatory Parity Dashboard")
st.markdown(
    """
This operational dashboard evaluates credit risk models across **three mitigation phases**:
1. **Phase 1 (Baseline):** Unmitigated models trained on imbalanced data.
2. **Phase 2 (SMOTE):** Data-level oversampling to balance default classes.
3. **Phase 3 (In/Post-Processing):** Dual-balance sample reweighting + fairness-constrained threshold optimization ($t$).
"""
)

tab1, tab2, tab3 = st.tabs([
    "🧪 Live Scoring & Subgroup Audit",
    "📊 Global Performance & Fairness Benchmarks",
    "📘 Dataset & Methodology Insights",
])

# -----------------------------------------------------------------------------
# TAB 1: LIVE CREDIT SCORING & INFERENCE
# -----------------------------------------------------------------------------
with tab1:
  st.header("Applicant Assessment & Multi-Phase Prediction")

  col_input, col_profile = st.columns([1, 2])

  test_df = (
      artifacts.get("test_df", pd.DataFrame())
      if artifacts
      else pd.DataFrame()
  )

  if input_mode == "Test Set Sample Mode" and not test_df.empty:
    with col_input:
      st.subheader("🎲 Test Set Sampler")
      if st.button(
          "Fetch Random Test Applicant", type="primary", use_container_width=True
      ):
        st.session_state["sample_idx"] = np.random.randint(0, len(test_df))

      if "sample_idx" not in st.session_state:
        st.session_state["sample_idx"] = 0

      idx = st.session_state["sample_idx"]
      sample_row = test_df.iloc[[idx]].copy()

      actual_target = int(sample_row["Actual_Default"].values[0])
      demo_group = (
          str(sample_row["Age_Group"].values[0])
          if "Age_Group" in sample_row
          else "Unknown"
      )

      # Extract feature subset (drop metadata columns if present)
      feature_cols = [
          c
          for c in test_df.columns
          if c not in ["Actual_Default", "Age_Group"]
      ]
      input_features = sample_row[feature_cols].copy()

    with col_profile:
      st.subheader("📋 Selected Applicant Card")
      p1, p2, p3 = st.columns(3)

      age_val = (
          input_features["Age"].values[0] if "Age" in input_features else "N/A"
      )
      income_val = (
          input_features["Income"].values[0]
          if "Income" in input_features
          else 0
      )
      cs_val = (
          input_features["CreditScore"].values[0]
          if "CreditScore" in input_features
          else "N/A"
      )

      p1.metric("Age", f"{age_val} yrs" if age_val != "N/A" else "N/A")
      p2.metric("Annual Income", f"${income_val:,.0f}")
      p3.metric("Credit Score", cs_val)

      st.caption(
          f"**Demographic Segment:** `{demo_group}` | **Ground Truth Outcome:**"
          f" `{'Default (High Risk)' if actual_target == 1 else 'Non-Default (Creditworthy)'}`"
      )

  else:
    with col_input:
      st.subheader("📝 Manual Profile Entry")
      applicant_age = st.slider("Age", 18, 75, 26)
      applicant_income = st.number_input(
          "Annual Income ($)", 10000, 250000, 48000
      )
      applicant_cs = st.slider("Credit Score", 300, 850, 640)
      applicant_dti = st.slider("Debt-to-Income Ratio (%)", 0.0, 60.0, 28.0)
      demo_group = (
          "Young Adult (<=30)" if applicant_age <= 30 else "Senior (>55)"
      )

      # Build dummy feature vector initialized to zero
      if artifacts and "scaler" in artifacts and hasattr(artifacts["scaler"], "feature_names_in_"):
        expected_cols = artifacts["scaler"].feature_names_in_
      elif not test_df.empty:
        expected_cols = [
            c
            for c in test_df.columns
            if c not in ["Actual_Default", "Age_Group"]
        ]
      else:
        expected_cols = ["Age", "Income", "CreditScore", "DTI"]

      input_features = pd.DataFrame(
          np.zeros((1, len(expected_cols))), columns=expected_cols
      )

      # Populate present numeric sliders dynamically if matching column names exist
      for col, val in [
          ("Age", applicant_age),
          ("Income", applicant_income),
          ("CreditScore", applicant_cs),
          ("DTI", applicant_dti),
      ]:
        if col in input_features.columns:
          input_features[col] = val

    with col_profile:
      st.subheader("📋 Configured Profile")
      st.info(f"**Assigned Protected Group:** `{demo_group}`")

  st.markdown("---")
  st.subheader(f"Multi-Phase Decision Engine ({selected_arch})")

  # -----------------------------------------------------------------------------
  # REAL MODEL INFERENCE ENGINE
  # -----------------------------------------------------------------------------
  if artifacts and "scaler" in artifacts:
    scaler = artifacts["scaler"]

    # Align input features strictly to scaler expectations
    if hasattr(scaler, "feature_names_in_"):
      expected_features = scaler.feature_names_in_
      # Reindex: missing expected columns are filled with 0.0, unexpected columns dropped
      input_features = input_features.reindex(
          columns=expected_features, fill_value=0.0
      )

    p1_key = "rf_p1" if selected_arch == "Random Forest" else "xgb_p1"
    p2_key = "rf_p2" if selected_arch == "Random Forest" else "xgb_p2"
    p3_key = "rf_p3" if selected_arch == "Random Forest" else "xgb_p3"

    p3_threshold = 0.51 if selected_arch == "Random Forest" else 0.61

    scaled_inputs = scaler.transform(input_features)

    prob_p1 = float(artifacts[p1_key].predict_proba(scaled_inputs)[:, 1][0])
    prob_p2 = float(artifacts[p2_key].predict_proba(scaled_inputs)[:, 1][0])
    prob_p3 = float(artifacts[p3_key].predict_proba(scaled_inputs)[:, 1][0])

    dec_p1 = prob_p1 >= 0.50
    dec_p2 = prob_p2 >= 0.50
    dec_p3 = prob_p3 >= p3_threshold

    card_p1, card_p2, card_p3 = st.columns(3)

    with card_p1:
      st.markdown("#### Phase 1: Baseline")
      st.caption("Threshold: `t = 0.50`")
      st.metric("Predicted Default Prob", f"{prob_p1:.1%}")
      if dec_p1:
        st.error("❌ **DENIED (High Risk)**")
      else:
        st.success("✅ **APPROVED (Low Risk)**")

    with card_p2:
      st.markdown("#### Phase 2: SMOTE")
      st.caption("Threshold: `t = 0.50`")
      st.metric("Predicted Default Prob", f"{prob_p2:.1%}")
      if dec_p2:
        st.error("❌ **DENIED (High Risk)**")
      else:
        st.success("✅ **APPROVED (Low Risk)**")

    with card_p3:
      st.markdown("#### Phase 3: Reweighted + Tuned")
      st.caption(f"Optimized Cutoff: `t = {p3_threshold:.2f}`")
      st.metric("Predicted Default Prob", f"{prob_p3:.1%}")
      if dec_p3:
        st.error("❌ **DENIED (High Risk)**")
      else:
        st.success("✅ **APPROVED (Low Risk)**")

    if dec_p1 and not dec_p3:
      st.success(
          "🎉 **Disparity Mitigation Effect:** Phase 1 rejected this applicant,"
          " but **Phase 3 approved them** under fairness-optimized decision"
          " rules!"
      )
    elif not dec_p1 and dec_p3:
      st.warning(
          "⚠️ **Risk Realignment:** Phase 1 approved this profile, but Phase 3"
          " flagged it for default risk under recalibrated thresholds."
      )

# -----------------------------------------------------------------------------
# TAB 2: GLOBAL PERFORMANCE & FAIRNESS BENCHMARKS
# -----------------------------------------------------------------------------
with tab2:
  st.header("Empirical Audit Matrix Across Mitigation Phases")

  benchmark_df = pd.DataFrame([
      {
          "Phase": "Phase 1 (Baseline)",
          "Model": "Random Forest",
          "Accuracy": 0.8863,
          "Recall": 0.1404,
          "F1-Score": 0.2330,
          "Age Disparate Impact": 0.9312,
          "Subgroup F1 Gap": 0.3938,
      },
      {
          "Phase": "Phase 2 (SMOTE)",
          "Model": "Random Forest",
          "Accuracy": 0.8003,
          "Recall": 0.8062,
          "F1-Score": 0.4983,
          "Age Disparate Impact": 1.0644,
          "Subgroup F1 Gap": 0.1654,
      },
      {
          "Phase": "Phase 3 (Weighted + Tuned)",
          "Model": "Random Forest",
          "Accuracy": 0.8880,
          "Recall": 0.7463,
          "F1-Score": 0.6210,
          "Age Disparate Impact": 0.9997,
          "Subgroup F1 Gap": 0.1320,
      },
      {
          "Phase": "Phase 1 (Baseline)",
          "Model": "XGBoost",
          "Accuracy": 0.8896,
          "Recall": 0.2267,
          "F1-Score": 0.3356,
          "Age Disparate Impact": 0.9257,
          "Subgroup F1 Gap": 0.3599,
      },
      {
          "Phase": "Phase 2 (SMOTE)",
          "Model": "XGBoost",
          "Accuracy": 0.8678,
          "Recall": 0.7406,
          "F1-Score": 0.5796,
          "Age Disparate Impact": 0.9858,
          "Subgroup F1 Gap": 0.1057,
      },
      {
          "Phase": "Phase 3 (Weighted + Tuned)",
          "Model": "XGBoost",
          "Accuracy": 0.8844,
          "Recall": 0.7203,
          "F1-Score": 0.6053,
          "Age Disparate Impact": 0.9839,
          "Subgroup F1 Gap": 0.1330,
      },
  ])

  st.dataframe(benchmark_df, use_container_width=True)

  st.markdown("---")
  st.subheader("Key Findings")
  c1, c2 = st.columns(2)
  with c1:
    st.write(
        "**1. Preservation of Overall Accuracy:**\n"
        "SMOTE (Phase 2) degraded overall accuracy to 80.03% on Random Forest."
        " Phase 3 dual-reweighting restored overall accuracy to **88.80%** while"
        " maintaining high recall (**74.63%**)."
    )
  with c2:
    st.write(
        "**2. Demographic Parity Attainment:**\n"
        "Phase 3 Random Forest achieved an Age Disparate Impact of **0.9997**"
        " (virtually perfect 1.0 parity), completely satisfying the legal"
        " 4/5ths rule (0.80 - 1.25)."
    )

# -----------------------------------------------------------------------------
# TAB 3: DATASET & METHODOLOGY INSIGHTS
# -----------------------------------------------------------------------------
with tab3:
  st.header("Research Methodology Overview")
  st.markdown(
      """
### Dual-Balance Sample Reweighting Formulation

Sample weights $W_i$ balance class outcomes $y \\in \\{0, 1\\}$ and protected demographic groups $A \\in \\{\\text{Young}, \\text{Senior}\\}$ simultaneously:

$$W_{i} = \\frac{N}{K \\times N_{y, a}}$$

### Decision Boundary Threshold Optimization

Class predictions apply tuned cutoffs $t^*$ to class 1 probabilities $P(y=1|X)$:

$$\\hat{y} = \\mathbb{I}(P(y=1|X) \\ge t^*)$$

Where $t^*$ is selected via grid sweep to maximize global $F1$-score under regulatory parity constraints:

$$\\text{Disparate Impact} = \\frac{P(\\hat{y}=0 \\mid A = \\text{Young})}{P(\\hat{y}=0 \\mid A = \\text{Senior})} \\ge 0.80$$
"""
  )