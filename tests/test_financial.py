import hashlib
import json
from types import SimpleNamespace
import pytest
from backend.services.preprocessing import statements,document_metadata,INDEX_SCHEMA
from backend.services.context_verification import candidate_check,features,quantities,non_policy
from backend.services.contradiction_engine import decide_pair
from backend.services.nli_classifier import label_mapping

HIGH={"contradiction":.99,"entailment":.005,"neutral":.005}
AGREE={"contradiction":.005,"entailment":.99,"neutral":.005}

def claim(text,**metadata):
    return dict(text=text,parent=text,id=hashlib.sha256(text.encode()).hexdigest(),document_id="a",filename="a.txt",page=1,**metadata)


def test_screenshot_pair_filtered_before_model():
    a=claim("Internal policy reference \u2022 Document 2 of 2 \u2022 Version 1.5 \u2022 Effective 1 October 2026")
    b=claim("NovaPay is a fictional organization; no policy below describes a real financial institution.")
    assert not candidate_check(a,b)[0]
    assert decide_pair(a,b,HIGH,HIGH)["classification"]=="Neutral"
    with pytest.raises(ValueError,match="No meaningful policy claims"):
        statements([(1,a["text"]+"\n\n"+b["text"])],"doc","metadata.txt")


def test_sections_versions_source_evidence_and_policy_words():
    text="Version 2.1\n\nEffective 1 October 2026\n\nSection 4.2 - Daily Transfer Limit: Customers may transfer up to INR 50,000 per day.\n\nThe refund policy becomes effective after identity verification."
    rows=statements([(3,text)],"doc","finance.pdf")
    assert rows[0]["text"]=="Customers may transfer up to INR 50,000 per day."
    assert rows[0]["section"]=="4.2" and rows[0]["version"]=="2.1"
    assert rows[0]["effective_date"]=="1 October 2026"
    assert rows[0]["text"] in rows[0]["parent"]
    assert any("policy becomes effective" in r["text"] for r in rows)
    assert all(r["index_schema"]==INDEX_SCHEMA for r in rows)


@pytest.mark.parametrize("a,b,expected",[
    ("The daily transfer limit for standard accounts is INR 50,000.","The daily transfer limit for standard accounts is INR 20,000.","Contradiction"),
    ("Basic account holders may withdraw INR 20,000 daily.","Premium account holders may withdraw INR 50,000 daily.","Needs Review"),
    ("Refunds are processed within five business days.","Refunds are processed within ten business days.","Needs Review"),
    ("Refunds take exactly five business days.","Refunds take exactly five calendar days.","Needs Review"),
    ("The daily transfer limit is INR 50,000.","Customer support operates on weekdays.","Neutral"),
    ("Customers may withdraw INR 20,000 daily.","The annual account maintenance fee is INR 500.","Neutral"),
    ("International card payments are permitted.","International card payments are prohibited.","Contradiction"),
    ("The annual interest rate for standard accounts is 4%.","The annual interest rate for standard accounts is 6%.","Contradiction"),
    ("The refund processing period was seven days in 2024.","The refund processing period is three days in 2026.","Policy Change"),
    ("The maintenance fee is USD 10 monthly.","The maintenance fee is INR 800 monthly.","Needs Review"),
])
def test_financial_scope_decisions(a,b,expected):
    result=decide_pair(claim(a),claim(b),HIGH,HIGH)
    assert result["classification"]==expected,result
    assert result["probabilities"]["contradiction"]==.99


def test_agreement_and_paraphrase_same_attribute():
    a=claim("Customers may withdraw INR 20,000 daily.")
    b=claim("The daily cash withdrawal ceiling is INR 20,000.")
    assert candidate_check(a,b)[0]
    assert features(a)["topic"]==features(b)["topic"]=="withdrawal limit"
    assert decide_pair(a,b,AGREE,AGREE)["classification"]=="Agreement"


def test_version_changes_not_unconditional_conflicts():
    a=claim("The daily transfer limit is INR 50,000.",version="1.0",organization="CedarBank")
    b=claim("The daily transfer limit is INR 20,000.",version="2.0",organization="CedarBank")
    for mode in ("strict","differences"):
        result=decide_pair(a,b,HIGH,HIGH,mode=mode)
        assert result["classification"]=="Policy Change"
        assert "1.0" in result["explanation"] and "2.0" in result["explanation"]
    unchanged=claim(a["text"],version="2.0",organization="CedarBank")
    assert decide_pair(a,unchanged,AGREE,AGREE)["classification"]=="Agreement"


def test_similarity_cutoff_and_unassessed_scope():
    a=claim("The daily transfer limit is INR 50,000.")
    assert not candidate_check(a,a,similarity=.1,min_similarity=.4)[0]
    different=claim(a["text"],organization="Other Bank")
    result=decide_pair(a,different)
    assert result["classification"]=="Needs Review" and result["probabilities"] is None
    assert not result["nli_evaluated"]


def test_rupees_indian_grouping_percent_and_working_days():
    q=quantities("The premium rate is 4.5% and the limit is \u20b91,00,000 per day; refunds take five working days.")
    assert any(x["value"]==100000 and x["currency"]=="INR" for x in q)
    assert any(x["value"]==4.5 and x["unit"]=="percent" for x in q)
    assert any(x["value"]==5 and x["unit"]=="business day" for x in q)


def test_inverse_mapping_must_agree():
    with pytest.raises(ValueError,match="label2id"):
        label_mapping(SimpleNamespace(id2label={0:"contradiction",1:"entailment",2:"neutral"},label2id={"contradiction":1,"entailment":0,"neutral":2}))


def test_repeated_pdf_furniture_removed_but_claims_preserved():
    pages=[(1,"ACME | Customer Finance Operations\n\n1. Refund Processing\nRefunds take exactly five business days.\n\nPage 1"),
        (2,"ACME | Customer Finance Operations\n\n2. Withdrawal Limit\nThe daily withdrawal limit is INR 10,000.\n\nPage 2")]
    rows=statements(pages,"doc","sample.pdf")
    assert len(rows)==2 and {r["page"] for r in rows}=={1,2}
    assert all(not non_policy(r["text"]) for r in rows)


def test_rejected_pairs_do_not_run_nli(monkeypatch):
    from backend.services import contradiction_engine as engine
    a=claim("Basic account holders may withdraw INR 20,000 daily.")
    b=claim("Premium account holders may withdraw INR 50,000 daily.");b["similarity"]=.9
    def candidates(*args,**kwargs):
        args[4].update(retrieved_pairs=1,rejected_pairs=0,rejections=[])
        return [(a,b)]
    monkeypatch.setattr(engine,"candidates",candidates)
    monkeypatch.setattr(engine,"predict",lambda *a,**k:pytest.fail("NLI must not run for mismatched categories"))
    result=engine.analyze(["a","b"])
    assert result["nli_pairs"]==0 and result["results"][0]["probabilities"] is None


def test_source_preserving_rebuild_and_failure_rollback(monkeypatch):
    from backend.core import storage
    from backend.core.config import DATA
    from backend.services import vector_store
    monkeypatch.setattr(vector_store,"encode",lambda texts:[[1.0]+[0.0]*383 for _ in texts])
    content=b"Internal policy reference - Version 1.0\n\nThe daily transfer limit is INR 50,000."
    did=hashlib.sha256(content).hexdigest()
    source=DATA/"uploads"/(did+".txt");source.write_bytes(content)
    storage.save("documents",did,dict(id=did,filename="rebuild.txt",filetype=".txt",statements=2,preview=content.decode()))
    previous=vector_store.collection()
    result=vector_store.rebuild()
    assert result["source_files_preserved"] and source.read_bytes()==content
    assert storage.get("documents",did)["index_schema"]==INDEX_SCHEMA
    assert vector_store.collection().count()==1
    active=(DATA/"index_manifest.json").read_bytes()
    original=source.read_bytes();source.write_bytes(b"changed source")
    with pytest.raises(ValueError,match="hash mismatch"):vector_store.rebuild()
    assert (DATA/"index_manifest.json").read_bytes()==active
    assert vector_store.collection().count()==1
    source.write_bytes(original)
    vector_store.collection().delete(where={"document_id":did});storage.delete("documents",did)
    source.unlink()


def test_stale_legacy_collection_is_not_silently_reused(tmp_path,monkeypatch):
    from backend.services import vector_store
    monkeypatch.setattr(vector_store,"DATA",tmp_path)
    vector_store.collection.cache_clear()
    old=vector_store.client().create_collection("statements",embedding_function=None)
    old.add(ids=["old"],documents=["Header only"],embeddings=[[1.0,0.0]],metadatas=[{"document_id":"old"}])
    assert vector_store.index_status()["requires_rebuild"]
    with pytest.raises(ValueError,match="old preprocessing schema"):vector_store.collection()
    vector_store.collection.cache_clear()


def test_numeric_inclusive_exclusive_overlap():
    from backend.services.context_verification import numeric_interval,intervals_overlap
    assert intervals_overlap((5,None,True,False),(None,5,False,True))
    assert not intervals_overlap((5,None,False,False),(None,5,False,True))
    a=claim("Refunds take at least five business days.")
    b=claim("Refunds take within ten business days.")
    assert decide_pair(a,b,HIGH,HIGH)["classification"]=="Needs Review"
    a=claim("Refunds take between three and seven business days.")
    b=claim("Refunds take between five and nine business days.")
    assert decide_pair(a,b,HIGH,HIGH)["classification"]=="Needs Review"


def test_report_with_scope_skip_has_no_invented_probability():
    from backend.services.report_generator import generate
    from backend.core.config import NLI_ID,EMBED_ID
    a=claim("Basic account holders may withdraw INR 20,000 daily.")
    b=claim("Premium account holders may withdraw INR 50,000 daily.")
    result=decide_pair(a,b)
    report=dict(results=[result],timestamp="2026-10-31",model=NLI_ID,embedding_model=EMBED_ID,candidate_pairs=1,threshold=.75,top_k=5)
    for format in ("csv","json","pdf"):
        content,mime=generate(report,format)
        assert content
    assert result["probabilities"] is None
