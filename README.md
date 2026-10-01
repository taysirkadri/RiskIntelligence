# RiskIntelligence

A small, runnable credit-risk workflow with a deterministic synthetic fallback (or CSV input).

## Run locally

```powershell
python -m pip install -e ".[api,ui,test]"
credit-risk-train
uvicorn credit_risk.api:app --reload       # POST /score, GET /health
streamlit run src/credit_risk/streamlit_app.py
pytest
```

The training command writes `outputs/credit_risk_model.joblib` and `outputs/metrics.json`.
It performs leakage-safe train/validation/test splits, preprocessing, logistic baseline,
LightGBM when installed (otherwise a sklearn gradient-boosting fallback), sigmoid
calibration, ROC/PR/Brier metrics, cost-sensitive threshold selection, group slices,
and SHAP when installed (otherwise permutation importance). Set `--data path.csv` for a
CSV containing the documented feature names and a binary `default` column.

This is an educational decision-support demo, not a lending approval system. Review
model card and governance guidance in `docs/model_card.md` before adapting it.
