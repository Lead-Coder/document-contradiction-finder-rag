from pathlib import Path
import requests
from streamlit.testing.v1 import AppTest

def test_all_workspace_pages_render(monkeypatch):
    class Response:
        ok=True
        def json(self):return []
    monkeypatch.setattr(requests,"request",lambda *a,**k:Response())
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/"frontend"/"app.py"),default_timeout=30).run()
    assert not app.exception
    for page in ["Upload Documents","Contradiction Finder","Ask About Documents","Evaluation Dashboard","Export Reports"]:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception,page


def test_populated_scope_and_policy_change_cards(monkeypatch):
    class Response:
        ok=True
        def json(self):return []
    monkeypatch.setattr(requests,"request",lambda *a,**k:Response())
    a=dict(filename="a.txt",page=1,text="The fee is INR 100.",parent="The fee is INR 100.",section="2")
    b=dict(filename="b.txt",page=2,text="The fee is INR 200.",parent="The fee is INR 200.",section="2")
    row=dict(a=a,b=b,classification="Needs Review",topic="account maintenance fee",probabilities=None,
        explanation="Customer categories differ; no unconditional conflict is established.",context_verification=dict(compatible=False,warnings=["Categories differ"],policy_changes=[]))
    change=dict(row,classification="Policy Change",explanation="Different policy versions.")
    result=dict(analysis_version="financial-v2",mode="differences",candidate_pairs=2,results=[row,change])
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/"frontend"/"app.py"),default_timeout=30)
    app.session_state["analysis_result"]=result
    app.run();app.sidebar.radio[0].set_value("Contradiction Finder").run()
    assert not app.exception
