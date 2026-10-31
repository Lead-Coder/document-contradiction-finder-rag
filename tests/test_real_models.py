import os
import pytest

@pytest.mark.models
@pytest.mark.skipif(os.getenv("DOCU_TEST_MODELS")!="1",reason="Run with DOCU_TEST_MODELS=1 after downloading model weights")
def test_real_inference_and_retrieval():
    from backend.services.nli_classifier import predict,load
    from backend.services import vector_store
    from backend.services.preprocessing import statements
    from backend.services.contradiction_engine import analyze
    pairs=[("Students are allowed to submit assignments after the deadline.","Students are not allowed to submit assignments after the deadline."),
        ("Employees receive 20 days of annual leave.","Employees receive 20 days of annual leave.")]
    probabilities=predict(pairs)
    assert all(abs(sum(p.values())-1)<1e-5 for p in probabilities)
    assert max(probabilities[0],key=probabilities[0].get)=="contradiction"
    assert max(probabilities[1],key=probabilities[1].get)=="entailment"
    for did,text in (("real-a",pairs[0][0]),("real-b",pairs[0][1])):
        vector_store.add(statements([(1,text)],did,did+".txt"))
    result=analyze(["real-a","real-b"])
    assert result["candidate_pairs"]==1
    assert result["results"][0]["classification"]=="Contradiction"
    for did in ("real-a","real-b"):vector_store.collection().delete(where={"document_id":did})

    from fastapi.testclient import TestClient
    from backend.main import app
    import time
    with TestClient(app) as client:
        ids=[]
        for name,text in (("live_a.txt",pairs[0][0]),("live_b.txt",pairs[0][1])):
            response=client.post("/documents/upload",files=[("files",(name,text,"text/plain"))])
            item=response.json()["files"][0]
            assert "error" not in item,item
            ids.append(item["document"]["id"])
        job=client.post("/analysis/compare",json={"document_ids":ids}).json()
        for _ in range(300):
            completed=client.get("/analysis/results/"+job["id"]).json()
            if completed["status"] in ("complete","failed"):break
            time.sleep(.1)
        assert completed["status"]=="complete",completed
        assert completed["result"]["results"][0]["classification"]=="Contradiction"
        assert completed["result"]["results"][0]["a"]["filename"] in ("live_a.txt","live_b.txt")
        assert client.get(f'/reports/{job["id"]}?format=pdf').content.startswith(b"%PDF")
        answer=client.post("/ask",json={"question":"Can students submit assignments after the deadline?"}).json()
        assert answer["evidence"] and answer["conflicts"]
        for did in ids:assert client.delete("/documents/"+did).status_code==200


@pytest.mark.models
@pytest.mark.skipif(os.getenv("DOCU_TEST_MODELS")!="1",reason="Requires cached real models")
def test_existing_novapay_pdfs_end_to_end():
    import sqlite3,json,hashlib
    from backend.core.config import ROOT
    from backend.services.document_loader import extract
    from backend.services.preprocessing import statements
    from backend.services.context_verification import non_policy
    from backend.services import vector_store
    from backend.services.contradiction_engine import analyze
    registry=ROOT/"data"/"registry.sqlite3"
    if not registry.exists():pytest.skip("Existing NovaPay documents not available")
    with sqlite3.connect(registry.resolve().as_uri()+"?mode=ro",uri=True) as db:
        docs=[json.loads(row[0]) for row in db.execute("SELECT payload FROM documents")]
    docs=[d for d in docs if "novapay" in d["filename"].lower() and d["filetype"]==".pdf"]
    if len(docs)!=2:pytest.skip("Exactly two existing NovaPay PDFs are required")
    hashes={};ids=[]
    for doc in docs:
        source=ROOT/"data"/"uploads"/(doc["id"]+doc["filetype"])
        content=source.read_bytes();hashes[source]=hashlib.sha256(content).hexdigest()
        rows=statements(extract(content,doc["filename"]),doc["id"],doc["filename"])
        assert all(not non_policy(r["text"]) for r in rows)
        assert all(r["section"] and r["version"] for r in rows)
        vector_store.add(rows);ids.append(doc["id"])
    try:
        result=analyze(ids,threshold=.95,mode="differences")
        assert any(r["classification"]=="Policy Change" for r in result["results"])
        assert all(r["classification"]!="Contradiction" for r in result["results"])
        assert result["rejected_pairs"]>0
        assert all(hashlib.sha256(path.read_bytes()).hexdigest()==value for path,value in hashes.items())
    finally:
        for did in ids:vector_store.collection().delete(where={"document_id":did})
