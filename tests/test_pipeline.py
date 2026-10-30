from types import SimpleNamespace
import pytest
from backend.services.document_loader import extract
from backend.services.preprocessing import statements
from backend.services.context_verification import verify
from backend.services.nli_classifier import label_mapping
from backend.services.contradiction_engine import classify

@pytest.mark.parametrize("name,content",[("empty.txt",b""),("bad.xyz",b"hello"),("bad.pdf",b"broken")])
def test_invalid_documents(name,content):
    with pytest.raises(ValueError):extract(content,name)

def test_deduplicate_preserve_evidence():
    rows=statements([(2,"Employees receive annual leave. Employees receive annual leave.")],"doc","policy.pdf")
    assert len(rows)==1
    assert rows[0]["page"]==2
    assert rows[0]["text"] in rows[0]["parent"]

def test_label_mapping_rejects_generic_labels():
    with pytest.raises(ValueError):label_mapping(SimpleNamespace(id2label={0:"LABEL_0",1:"LABEL_1",2:"LABEL_2"}))
    assert label_mapping(SimpleNamespace(id2label={0:"neutral",1:"contradiction",2:"entailment"}))[1]=="contradiction"

@pytest.mark.parametrize("left,right",[
    ("The office is closed on January 1, 2026.","The office is open on January 2, 2026."),
    ("Fees were 500 in 2024.","Fees were 700 in 2026."),
    ("Department A has a large budget.","Department B has a smaller budget."),
    ("Students get refunds if they cancel early.","Students get refunds if they cancel late.")])
def test_scope_abstention(left,right):
    reasons=verify(dict(text=left),dict(text=right))
    assert reasons
    p=dict(contradiction=.96,entailment=.02,neutral=.02)
    assert classify(p,p,.75,reasons)[0]=="Needs Review"

def test_numeric_difference_is_not_a_rule():
    reasons=verify(dict(text="The upload limit is 20 MB."),dict(text="The upload limit is 10 MB."))
    neutral=dict(contradiction=.1,entailment=.1,neutral=.8)
    assert classify(neutral,neutral,.75,reasons)[0]=="Neutral"

def test_low_confidence_and_directional_disagreement():
    low=dict(contradiction=.5,entailment=.3,neutral=.2)
    strong=dict(contradiction=.95,entailment=.03,neutral=.02)
    other=dict(contradiction=.02,entailment=.96,neutral=.02)
    assert classify(low,low,.75,[])[0]=="Needs Review"
    assert classify(strong,other,.75,[])[0]=="Needs Review"

def test_docx_and_pdf(tmp_path):
    from docx import Document
    import pymupdf
    doc=Document();doc.add_paragraph("Employees receive annual leave.")
    path=tmp_path/"sample.docx";doc.save(path)
    assert "annual leave" in extract(path.read_bytes(),path.name)[0][1]
    pdf=pymupdf.open();page=pdf.new_page();page.insert_text((50,50),"Employees receive annual leave.")
    pages=extract(pdf.tobytes(),"sample.pdf")
    assert pages[0][0]==1 and "annual leave" in pages[0][1]
    blank=pymupdf.open();blank.new_page()
    with pytest.raises(ValueError,match="OCR"):extract(blank.tobytes(),"scanned.pdf")

def test_unrelated_subjects_and_compatible_bounds_abstain():
    p=dict(contradiction=.999,entailment=.0005,neutral=.0005)
    examples=[("The park has wooden benches.","The pharmacy sells prescription medicines."),
        ("The service requires protocol version 4 or higher.","The service requires protocol version 5 or higher.")]
    for a,b in examples:
        reasons=verify(dict(text=a),dict(text=b))
        assert reasons
        assert classify(p,p,.75,reasons)[0]=="Needs Review"
