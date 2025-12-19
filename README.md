# NAFLD Risk Predictor and Analyzer

A small Streamlit app and model notebook for predicting NAFLD (Non-Alcoholic Fatty Liver Disease) risk from anthropometric and lab features. The project includes a cleaned dataset, a training notebook, and a Streamlit UI that loads prebuilt model artifacts.

**Files**
- [app.py](app.py): Streamlit application that loads `nafld_artifacts.pkl` to predict NAFLD risk from user inputs.
- [nafld_artifacts.pkl](nafld_artifacts.pkl): Pickled artifact bundle (model, scaler, encoders, feature list) required by `app.py`.
- [Project_Model.ipynb](Project_Model.ipynb): Jupyter notebook with exploratory analysis and model development (cells present; not executed in the repo snapshot).
- [Cleaned_NAFLD_Dataset.csv](Cleaned_NAFLD_Dataset.csv): Cleaned dataset used for analysis and model training. Key columns: `RIDAGEYR, RIAGENDR, RIDRETH1, BMXBMI, BMXWAIST, BMXWT, BMXHT, BPXSY1, BPXDI1, LBXTC, LBDHDD, Waist_Height_Ratio`.

**Overview**
- The Streamlit app loads an artifact bundle (`nafld_artifacts.pkl`) containing:
  - `model_name`, `model` (sklearn-compatible), `scaler`, `label_encoders`, and `features`.
- The UI collects age, gender, ethnicity, height, weight, BMI, waist circumference, blood pressure and lipid values, computes waist-height ratio, preprocesses inputs using the saved encoders/scaler, and shows a predicted NAFLD risk probability and heuristic risk indicators.

**Requirements**
- Python 3.9+ recommended
- Primary packages:
  - streamlit
  - numpy
  - pandas
  - scikit-learn

Install quickly with:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install streamlit numpy pandas scikit-learn
```

(Or create a `requirements.txt` with the packages above and run `pip install -r requirements.txt`.)

**Run the app**
1. Ensure `nafld_artifacts.pkl` is in the same folder as `app.py`.
2. Start the Streamlit UI:

```bash
streamlit run app.py
```

Open the provided URL in your browser to interact with the predictor.

**Notes & Recommendations**
- This tool is for educational purposes only and is not medical advice. Interpret results with caution and consult healthcare professionals for clinical decisions.
- `Project_Model.ipynb` contains the modeling pipeline — explore the notebook to review feature engineering, model selection, evaluation, and artifact creation.
- If you want reproducible installs, consider adding a `requirements.txt` or `pyproject.toml`.

**Contact / Next steps**
- To expand: add input validation, model explainability (SHAP), unit tests, and CI.

---
Generated on December 19, 2025
