# NYC Airbnb End-to-End Data Science Pipeline

This project delivers a full pipeline for the NYC Airbnb Open Data dataset (`AB_NYC_2019.csv`):
- exploratory data analysis (EDA)
- cross-validated model benchmarking
- holdout evaluation
- SHAP explainability for every trained model
- Streamlit dashboard with an executive summary and interactive feature playground

## Models
The training pipeline includes:
- Linear Regression
- Lasso Regression
- Ridge Regression
- CART (Decision Tree Regressor)
- Random Forest Regressor
- LightGBM Regressor (if `lightgbm` is installed)

## Run
From the `archive/` folder:

```bash
pip install -r requirements.txt
python build_artifacts.py
streamlit run app.py
```

## Generated Artifacts
Saved under `archive/artifacts/`:
- `models/*.joblib`: fitted model pipelines
- `metrics.json`: CV means/std/fold scores + holdout metrics
- `summary.json`: run metadata and best model
- `diagnostics.json`: residual diagnostics payload
- `shap_index.json`: pointers to SHAP files per model
- `shap/<model_name>/...`: SHAP arrays, transformed data, and importance table
- `plots/shap_*.png`: summary and bar SHAP plots
