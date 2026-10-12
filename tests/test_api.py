import time

from fastapi.testclient import TestClient

from deepscout.api.app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_404_for_unknown_run():
    assert client.get("/api/research/nope").status_code == 404


def test_mock_research_via_api():
    r = client.post("/api/research", json={"question": "大模型长上下文技术如何扩展？", "mock": True})
    assert r.status_code == 200
    run_id = r.json()["run_id"]

    status = {}
    for _ in range(120):  # 最多等 60s
        status = client.get(f"/api/research/{run_id}").json()
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.5)
    assert status["status"] == "done", status

    report = client.get(f"/api/research/{run_id}/report").json()
    assert "[1]" in report["report_md"]
    assert report["stats"]["llm_calls"] >= 6
