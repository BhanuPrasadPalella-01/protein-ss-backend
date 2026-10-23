"""API tests: run with `pytest` from the repository root."""
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health_get_and_head():
    assert client.get("/health").text == "ok"
    assert client.head("/health").status_code == 200


def test_predict_returns_one_label_per_residue():
    seq = "MKTAYIAKQRQISFVKSHFSRQ"
    r = client.post("/predict/baseline", json={"sequence": seq})
    assert r.status_code == 200
    body = r.json()
    assert body["sequence"] == seq
    assert len(body["predicted_structure"]) == len(seq)
    assert set(body["predicted_structure"]) <= {"H", "E", "C"}
    assert abs(sum(body["percentages"].values()) - 100) < 0.5


def test_predict_cleans_case_and_whitespace():
    r = client.post("/predict/baseline", json={"sequence": "  mktayiakqr  "})
    assert r.status_code == 200
    assert r.json()["sequence"] == "MKTAYIAKQR"


def test_predict_rejects_bad_input():
    assert client.post("/predict/baseline", json={"sequence": ""}).status_code == 400
    bad = client.post("/predict/baseline", json={"sequence": "MKTB1"})
    assert bad.status_code == 400
    assert "'1'" in bad.json()["detail"] and "'B'" in bad.json()["detail"]
    assert client.post("/predict/baseline", json={"sequence": "A" * 701}).status_code == 400


def test_samples_and_sample_prediction():
    samples = client.get("/samples").json()
    assert len(samples) > 0
    first = samples[0]
    assert len(first["sequence"]) == len(first["true_structure"]) == first["length"]
    pred = client.get(f"/samples/{first['id']}/predict").json()
    assert len(pred["pssm_prediction"]) == len(pred["attention_prediction"]) == first["length"]
    assert 0 <= pred["attention_match_accuracy"] <= 100
    assert client.get("/samples/NOPE/predict").status_code == 404


def test_comparison_lists_three_models_in_order():
    models = client.get("/comparison").json()["models"]
    assert [m["name"] for m in models] == ["Baseline", "+PSSM", "+Attention"]
    q3 = [m["q3_accuracy"] for m in models]
    assert q3 == sorted(q3)
