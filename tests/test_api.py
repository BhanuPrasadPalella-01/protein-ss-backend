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


def test_trace_matches_the_model_and_has_consistent_shapes():
    import numpy as np
    import torch
    from main import SAMPLES, aa_to_idx, attention_model

    sample = next(iter(SAMPLES.values()))
    body = client.get(f"/samples/{sample['id']}/trace").json()
    w = body["window"]
    L, start = body["length"], body["start"]
    assert L == 12 and 0 <= start <= sample["length"] - L
    assert len(w["residues"]) == len(w["true"]) == len(w["predicted"]) == L
    assert np.array(w["pssm"]).shape == (L, 22)
    assert np.array(w["embedding"]).shape == (L, 32)
    assert np.array(w["bilstm"]).shape == (L, 128)
    assert np.array(w["attention"]).shape == (4, L, L)
    # Each query's weights over the whole protein sum to 1: inside the window plus outside it.
    totals = np.array(w["attention"]).sum(axis=-1) + np.array(w["attention_outside"])
    assert np.allclose(totals, 1, atol=0.01)

    # Running the layers one at a time must give the same output as the model's own forward pass.
    X = torch.tensor([[aa_to_idx.get(c, aa_to_idx["X"]) for c in sample["sequence"]]])
    P = torch.tensor(np.array([sample["pssm"]], dtype=np.float32))
    with torch.no_grad():
        expected = torch.softmax(attention_model(X, P), dim=-1)[0, start:start + L].numpy()
    assert np.allclose(np.array(w["probabilities"]), expected, atol=0.002)
    assert body["predicted_structure"] == client.get(f"/samples/{sample['id']}/predict").json()["attention_prediction"]


def test_trace_validates_parameters():
    sample_id = client.get("/samples").json()[0]["id"]
    assert client.get("/samples/NOPE/trace").status_code == 404
    assert client.get(f"/samples/{sample_id}/trace?start=-1").status_code == 400
    assert client.get(f"/samples/{sample_id}/trace?start=100000").status_code == 400
    assert client.get(f"/samples/{sample_id}/trace?length=2").status_code == 400
    assert client.get(f"/samples/{sample_id}/trace?start=0&length=8").json()["window"]["residues"] == \
        client.get("/samples").json()[0]["sequence"][:8]
