import json
from pathlib import Path
import pytest
from credit_risk.pipeline import make_synthetic_data, eda_summary, train_demo, choose_cost_threshold

def test_synthetic_data_is_deterministic_and_valid():
    a, b = make_synthetic_data(100), make_synthetic_data(100)
    assert a.equals(b)
    assert set(a.default.unique()) <= {0, 1}
    assert 0 < a.default.mean() < 1

def test_cost_threshold_responds_to_fn_cost():
    threshold, cost = choose_cost_threshold([0, 0, 1, 1], [.1, .4, .6, .9], fn_cost=10)
    assert 0.05 <= threshold <= .95 and cost["fn_cost"] == 10

def test_training_writes_metrics(tmp_path):
    result = train_demo(output_dir=tmp_path)
    assert result["selected_model"] in result["models"]
    assert (tmp_path / "credit_risk_model.joblib").exists()
    payload = json.loads((tmp_path / "metrics.json").read_text())
    assert payload["splits"]["train"] > payload["splits"]["validation"]
    assert "roc_auc" in payload["models"][payload["selected_model"]]["test"]
