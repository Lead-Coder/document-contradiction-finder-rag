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
