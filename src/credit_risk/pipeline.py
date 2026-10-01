"""Data, preprocessing, models, metrics, and governance for a small credit demo."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

RANDOM_STATE = 42
TARGET = "default"
GROUP = "age_group"
FEATURES = ["income", "debt_ratio", "credit_history_years", "loan_amount", "num_late_payments", "employment_years", "age", GROUP, "home_ownership"]

def make_synthetic_data(n: int = 1800, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Create a deterministic, realistic-enough fallback dataset (1 means default)."""
    rng = np.random.default_rng(random_state)
    age = rng.integers(21, 76, n)
    income = np.clip(rng.lognormal(10.8, .55, n), 18000, 250000)
    debt_ratio = np.clip(rng.beta(2.2, 4.0, n) + rng.normal(0, .03, n), .01, .98)
    history = np.clip(age - rng.integers(18, 30, n), 0, 45)
    loan = np.clip(income * rng.uniform(.05, .65, n), 1000, 100000)
    late = rng.poisson(1.0 + 2.8 * debt_ratio, n)
    employment = np.clip(rng.gamma(2.5, 3, n), 0, 40)
    home = rng.choice(["rent", "own", "mortgage"], n, p=[.38, .25, .37])
    age_group = np.where(age < 30, "young", np.where(age < 55, "middle", "older"))
    logit = -3.2 + 3.2 * debt_ratio + .16 * late + .000006 * loan - .000004 * income - .035 * employment
    logit += (home == "rent") * .25 + (age_group == "young") * .18
    probability = 1 / (1 + np.exp(-logit))
    default = rng.binomial(1, probability)
    return pd.DataFrame({"income": income, "debt_ratio": debt_ratio, "credit_history_years": history,
        "loan_amount": loan, "num_late_payments": late, "employment_years": employment,
        "age": age, GROUP: age_group, "home_ownership": home, TARGET: default})

def load_data(path: str | os.PathLike[str] | None = None) -> pd.DataFrame:
    """Load a CSV when supplied; otherwise use the deterministic public-data fallback."""
    if path:
        frame = pd.read_csv(path)
        if TARGET not in frame.columns:
            raise ValueError(f"CSV must contain '{TARGET}'")
        return frame
    return make_synthetic_data()

def eda_summary(frame: pd.DataFrame) -> dict[str, Any]:
    numeric = frame.select_dtypes(include=np.number)
    return {"rows": int(len(frame)), "columns": list(frame.columns), "missing": frame.isna().sum().to_dict(),
            "target_rate": float(frame[TARGET].mean()), "numeric_describe": numeric.describe().round(4).to_dict()}

def _build_preprocessor(frame: pd.DataFrame):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    numerical = [c for c in FEATURES if c in frame.columns and pd.api.types.is_numeric_dtype(frame[c])]
    categorical = [c for c in FEATURES if c in frame.columns and c not in numerical]
    return ColumnTransformer([("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numerical),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)]), numerical, categorical

def _models():
    from sklearn.linear_model import LogisticRegression
    models = {"logistic": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)}
    try:
        from lightgbm import LGBMClassifier
        models["lightgbm"] = LGBMClassifier(n_estimators=180, learning_rate=.05, num_leaves=24, verbosity=-1, random_state=RANDOM_STATE)
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        models["lightgbm_fallback"] = GradientBoostingClassifier(n_estimators=60, learning_rate=.06, random_state=RANDOM_STATE)
    return models

def metrics(y_true, probabilities, threshold=.5) -> dict[str, float]:
    from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,
        precision_score, recall_score, roc_auc_score)
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    return {"roc_auc": float(roc_auc_score(y_true, probabilities)), "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)), "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)), "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)), "threshold": float(threshold)}

def choose_cost_threshold(y_true, probabilities, fp_cost=1.0, fn_cost=5.0) -> tuple[float, dict[str, float]]:
    """Choose threshold minimizing expected false-positive/false-negative cost."""
    y, p = np.asarray(y_true), np.asarray(probabilities)
    candidates = np.linspace(.05, .95, 91)
    costs = [fp_cost * np.sum((p >= t) & (y == 0)) + fn_cost * np.sum((p < t) & (y == 1)) for t in candidates]
    i = int(np.argmin(costs))
    return float(candidates[i]), {"fp_cost": fp_cost, "fn_cost": fn_cost, "expected_cost": float(costs[i])}

def fairness_slices(frame: pd.DataFrame, probabilities, threshold=.5, group=GROUP) -> dict[str, dict[str, float]]:
    """Report group-wise selection rate, default rate, and TPR; not a legal fairness conclusion."""
    from sklearn.metrics import recall_score
    out = {}
    pred = (np.asarray(probabilities) >= threshold).astype(int)
    for value, idx in frame.groupby(group, dropna=False).groups.items():
        y = frame.loc[idx, TARGET].to_numpy()
        out[str(value)] = {"n": int(len(idx)), "default_rate": float(y.mean()), "selection_rate": float(pred[np.asarray(frame.index.get_indexer(idx))].mean()),
                           "tpr": float(recall_score(y, pred[np.asarray(frame.index.get_indexer(idx))], zero_division=0))}
    return out

def feature_importance(model, X, y) -> dict[str, float]:
    """Use SHAP when installed, otherwise permutation importance."""
    try:
        import shap
        transformed = model.named_steps["preprocess"].transform(X)
        explainer = shap.Explainer(model.named_steps["model"], transformed)
        values = np.abs(explainer(transformed[: min(500, len(X))]).values).mean(axis=0)
        names = model.named_steps["preprocess"].get_feature_names_out()
        return {str(k): float(v) for k, v in sorted(zip(names, values), key=lambda x: -x[1])[:20]}
    except (ImportError, Exception):
        from sklearn.inspection import permutation_importance
        result = permutation_importance(model, X, y, n_repeats=1, random_state=RANDOM_STATE, scoring="roc_auc")
        return {str(k): float(v) for k, v in sorted(zip(X.columns, result.importances_mean), key=lambda x: -x[1])}

def train_demo(data_path=None, output_dir="outputs", random_state=RANDOM_STATE) -> dict[str, Any]:
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    frame = load_data(data_path)
    X, y = frame[FEATURES], frame[TARGET].astype(int)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=.4, stratify=y, random_state=random_state)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=.5, stratify=y_temp, random_state=random_state)
    preprocessor, _, _ = _build_preprocessor(frame)
    results = {"dataset": eda_summary(frame), "splits": {"train": len(X_train), "validation": len(X_valid), "test": len(X_test)}, "models": {}}
    best_name, best_model, best_auc = None, None, -1
    for name, estimator in _models().items():
        model = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        model.fit(X_train, y_train)
        valid_p = model.predict_proba(X_valid)[:, 1]
        threshold, costs = choose_cost_threshold(y_valid, valid_p)
        test_p = model.predict_proba(X_test)[:, 1]
        entry = {"validation": metrics(y_valid, valid_p, threshold), "test": metrics(y_test, test_p, threshold), "cost": costs,
                 "fairness": fairness_slices(frame.loc[X_test.index], test_p, threshold), "feature_importance": feature_importance(model, X_test, y_test)}
        results["models"][name] = entry
        if entry["validation"]["roc_auc"] > best_auc: best_name, best_model, best_auc = name, model, entry["validation"]["roc_auc"]
    # Calibrated baseline is explicitly evaluated and retained for serving.
    from sklearn.calibration import CalibratedClassifierCV
    calibrated = Pipeline([("preprocess", _build_preprocessor(frame)[0]), ("model", CalibratedClassifierCV(_models()["logistic"], cv=3, method="sigmoid"))])
    calibrated.fit(X_train, y_train)
    cal_p = calibrated.predict_proba(X_test)[:, 1]
    results["calibrated_logistic"] = {"test": metrics(y_test, cal_p), "brier": float(metrics(y_test, cal_p)["brier"])}
    results["selected_model"] = best_name
    results["model"] = best_model
    results["calibrated_model"] = calibrated
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(best_model, out / "credit_risk_model.joblib")
    (out / "metrics.json").write_text(json.dumps({k:v for k,v in results.items() if k not in ("model", "calibrated_model")}, indent=2, default=str), encoding="utf-8")
    return results

def score_application(application: dict[str, Any], model_path="outputs/credit_risk_model.joblib") -> dict[str, Any]:
    import joblib
    model = joblib.load(model_path)
    row = pd.DataFrame([application])
    probability = float(model.predict_proba(row[FEATURES])[:, 1][0])
    return {"default_probability": probability, "decision": "decline_or_review" if probability >= .5 else "approve", "threshold": .5}

def main():
    parser = argparse.ArgumentParser(description="Train the credit-risk demo")
    parser.add_argument("--data", help="CSV path; omitted uses synthetic fallback")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args(); result = train_demo(args.data, args.output_dir)
    print(json.dumps({"selected_model": result["selected_model"], "metrics": result["models"][result["selected_model"]]["test"]}, indent=2))

if __name__ == "__main__": main()

