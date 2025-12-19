import pickle
import numpy as np
import streamlit as st

# ---------- Load artifacts ----------
with open("nafld_artifacts.pkl", "rb") as f:
    bundle = pickle.load(f)

model_name = bundle["model_name"]
model = bundle["model"]
scaler = bundle["scaler"]
label_encoders = bundle["label_encoders"]  # {'RIAGENDR': LabelEncoder, 'RIDRETH1': LabelEncoder}
features = bundle["features"]

st.set_page_config(page_title="NAFLD Risk Predictor", layout="centered")
st.title("🩺 NAFLD Risk Predictor")
st.caption(f"Model: {model_name}")

# ---------- UI Inputs ----------
col1, col2 = st.columns(2)
with col1:
    RIDAGEYR = st.number_input("Age (years)", min_value=18, max_value=90, value=45)
    RIAGENDR = st.selectbox("Gender", options=label_encoders['RIAGENDR'].classes_.tolist())
    RIDRETH1 = st.selectbox("Ethnicity", options=label_encoders['RIDRETH1'].classes_.tolist())
    BMXHT = st.number_input("Height (cm)", min_value=120.0, max_value=210.0, value=170.0, step=0.1)
    BMXWT = st.number_input("Weight (kg)", min_value=35.0, max_value=180.0, value=78.0, step=0.1)

with col2:
    BMXBMI = st.number_input("BMI", min_value=15.0, max_value=60.0, value=27.0, step=0.1)
    BMXWAIST = st.number_input("Waist Circumference (cm)", min_value=50.0, max_value=170.0, value=100.0, step=0.1)
    BPXSY1 = st.number_input("Systolic BP", min_value=80.0, max_value=220.0, value=125.0, step=1.0)
    BPXDI1 = st.number_input("Diastolic BP", min_value=40.0, max_value=140.0, value=80.0, step=1.0)
    LBXTC = st.number_input("Total Cholesterol (mg/dL)", min_value=80.0, max_value=400.0, value=200.0, step=1.0)
    LBDHDD = st.number_input("HDL (mg/dL)", min_value=10.0, max_value=120.0, value=45.0, step=1.0)

# compute Waist_Height_Ratio (cm/cm)
Waist_Height_Ratio = BMXWAIST / BMXHT if BMXHT else 0.0

st.write(f"**Waist-Height Ratio:** {Waist_Height_Ratio:.3f}")

# ---------- Preprocess to model input ----------
# encode cats
RIAGENDR_enc = label_encoders['RIAGENDR'].transform([RIAGENDR])[0] if RIAGENDR in label_encoders['RIAGENDR'].classes_ else 0
RIDRETH1_enc = label_encoders['RIDRETH1'].transform([RIDRETH1])[0] if RIDRETH1 in label_encoders['RIDRETH1'].classes_ else 0

# strict ordering as used in training
row = {
    'RIDAGEYR': RIDAGEYR,
    'RIAGENDR': RIAGENDR_enc,
    'RIDRETH1': RIDRETH1_enc,
    'BMXBMI': BMXBMI,
    'BMXWAIST': BMXWAIST,
    'BMXWT': BMXWT,
    'BMXHT': BMXHT,
    'BPXSY1': BPXSY1,
    'BPXDI1': BPXDI1,
    'LBXTC': LBXTC,
    'LBDHDD': LBDHDD,
    'Waist_Height_Ratio': Waist_Height_Ratio
}

# reorder to features list
x = np.array([[row[f] for f in features]], dtype=float)
x_scaled = scaler.transform(x)

# ---------- Predict ----------
proba = model.predict_proba(x_scaled)[0,1]
pred = int(proba >= 0.5)

st.markdown("---")
st.subheader("Prediction")
st.metric("NAFLD Risk Probability", f"{proba:.2%}")
st.write("**Predicted Class:**", "1 (At Risk)" if pred == 1 else "0 (Low Risk)")

# ---------- Friendly guidance (not medical advice) ----------
st.info(
    "This tool is for educational purposes only and **not** medical advice. "
    "Discuss results with a healthcare professional."
)

# OPTIONAL: Simple risk factors echo
risk_factors = []
if BMXBMI >= 30: risk_factors.append("High BMI (≥30)")
if (RIAGENDR == 'Male' and BMXWAIST >= 102) or (RIAGENDR != 'Male' and BMXWAIST >= 88):
    risk_factors.append("Abdominal obesity (waist)")
if LBXTC > 200: risk_factors.append("High total cholesterol")
if (RIAGENDR == 'Male' and LBDHDD < 40) or (RIAGENDR != 'Male' and LBDHDD < 50):
    risk_factors.append("Low HDL")

if risk_factors:
    st.markdown("**Contributing risk indicators (heuristic):**")
    for rf in risk_factors:
        st.write(f"- {rf}")
