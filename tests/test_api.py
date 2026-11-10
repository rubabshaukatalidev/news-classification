from fastapi.testclient import TestClient

from backend.app import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert "sports" in r.json()["classes"]


def test_classify_sports():
    r = client.post("/api/classify", json={"text": "پاکستان نے کرکٹ میچ جیت لیا"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "sports"
    assert 0 < body["confidence"] <= 1
    assert abs(sum(s["score"] for s in body["scores"]) - 1) < 1e-3


def test_classify_economy():
    r = client.post("/api/classify", json={"text": "اسٹیٹ بینک نے شرح سود اور ڈالر کی قیمت پر بیان جاری کیا"})
    assert r.json()["label"] == "economy"


def test_blank_rejected():
    assert client.post("/api/classify", json={"text": "   "}).status_code == 422
    assert client.post("/api/classify", json={"text": ""}).status_code == 422


def test_batch():
    r = client.post("/api/classify/batch", json={"texts": ["ڈینگی کے مریض ہسپتال میں داخل", "میٹرک کا نتیجہ آج جاری ہوگا"]})
    assert r.status_code == 200
    labels = [x["label"] for x in r.json()["results"]]
    assert labels == ["health", "education"]


def test_frontend_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
