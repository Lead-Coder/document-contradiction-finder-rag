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
