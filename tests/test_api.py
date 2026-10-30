import time
from fastapi.testclient import TestClient
from backend.main import app
from backend.services import vector_store, nli_classifier

def test_api_workflow_with_explicit_test_doubles(monkeypatch):
    # Deterministic doubles test wiring only, never model quality. Real weights
    # are exercised separately by the opt-in models test.
    def fake_encode(texts):
        return [[1.0]+[0.0]*383 for _ in texts]
    monkeypatch.setattr(vector_store,"encode",fake_encode)
    import backend.services.contradiction_engine as engine
    monkeypatch.setattr(engine,"predict",lambda pairs,progress=None:[dict(contradiction=.95,entailment=.03,neutral=.02) for _ in pairs])
    with TestClient(app) as client:
        assert client.get("/health").status_code==200
        assert client.post("/analysis/compare",json={"document_ids":["x","x"]}).status_code==422
        uploaded=[]
        for filename,text in [("a.txt","The maximum upload size is 20 MB."),("b.txt","The maximum upload size is 10 MB.")]:
            response=client.post("/documents/upload",files=[("files",(filename,text,"text/plain"))])
            item=response.json()["files"][0]
            assert "error" not in item,item
            uploaded.append(item["document"]["id"])
        duplicate=client.post("/documents/upload",files=[("files",("a.txt","The maximum upload size is 20 MB.","text/plain"))])
        assert duplicate.json()["files"][0]["duplicate"]
        assert len(client.get("/documents").json())==2
        job=client.post("/analysis/compare",json={"document_ids":uploaded}).json()
        for _ in range(200):
            result=client.get("/analysis/results/"+job["id"]).json()
            if result["status"] in ("complete","failed"):break
            time.sleep(.02)
        assert result["status"]=="complete",result
        assert result["result"]["candidate_pairs"]==1
        assert result["result"]["results"][0]["classification"]=="Contradiction"
        for fmt in ("csv","json","pdf"):
            response=client.get(f'/reports/{job["id"]}?format={fmt}')
            assert response.status_code==200 and len(response.content)>50
            if fmt=="pdf":assert response.content.startswith(b"%PDF")
        assert client.delete("/documents/"+uploaded[0]).status_code==200
        assert len(client.get("/documents").json())==1
        assert client.delete("/documents").status_code==200
        assert client.get("/documents").json()==[]

def test_missing_results_and_invalid_inputs():
    with TestClient(app) as client:
        assert client.get("/analysis/results/missing").status_code==404
        assert client.post("/analysis/compare",json={"document_ids":["a","b"],"threshold":2}).status_code==422
        bad=client.post("/documents/upload",files=[("files",("bad.exe",b"invalid","application/octet-stream"))])
        assert "error" in bad.json()["files"][0]
