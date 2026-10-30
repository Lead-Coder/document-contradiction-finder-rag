import os
import difflib
import html
from collections import Counter
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
API=os.getenv("DOCU_API","http://127.0.0.1:8000").rstrip("/")
st.set_page_config(page_title="DocuConflict AI",layout="wide")
st.markdown("""<style>
.stApp {background:linear-gradient(135deg,#f6f8fc,#eef3f8);color:#15243b}
[data-testid="stSidebar"] {background:#13243c}
[data-testid="stSidebar"] * {color:#e8effa}
h1,h2,h3 {letter-spacing:-.035em}
[data-testid="stMetric"] {background:white;padding:1.1rem;border-radius:14px;border:1px solid #dce4ef}
.claim {background:#fff;padding:20px;border:1px solid #dce4ef;border-radius:12px;min-height:100px;color:#15243b}
mark {background:#fff0a6;padding:1px 2px}
</style>""",unsafe_allow_html=True)

# All retrieval, model inference, classification and evaluation live in FastAPI.
def api(method,path,**kwargs):
    try:
        response=requests.request(method,API+path,timeout=kwargs.pop("timeout",120),**kwargs)
        if not response.ok:
            try: detail=response.json().get("detail",response.text)
            except ValueError: detail=response.text
            st.error(str(detail));return None
        return response
    except requests.RequestException as exc:
        st.error(f"Backend unavailable: {exc}. Start FastAPI on {API}.")
        return None

def data(method,path,**kwargs):
    response=api(method,path,**kwargs)
    return response.json() if response is not None else None

def show_evaluation(result):
    if result["kind"]=="sentence_nli":
        cols=st.columns(4)
        for col,key in zip(cols,["accuracy","macro_precision","macro_recall","macro_f1"]):
            col.metric(key.replace("_"," ").title(),f'{result[key]:.3f}')
        st.plotly_chart(px.imshow(result["confusion_matrix"],x=result["labels"],y=result["labels"],text_auto=True,
            labels=dict(x="Predicted",y="Actual")),use_container_width=True)
        st.dataframe(pd.DataFrame(result["per_class"]).T,use_container_width=True)
        only_wrong=st.checkbox("Show misclassified examples only")
        examples=[r for r in result["examples"] if not only_wrong or not r["correct"]]
        st.dataframe(examples,use_container_width=True)
    else:
        st.info("Synthetic document benchmark. These measurements are separate from sentence-level NLI accuracy.")
        st.dataframe(result["experiments"],use_container_width=True)
        st.plotly_chart(px.bar(result["experiments"],x="top_k",y=["recall_at_k","precision","recall","f1"],barmode="group"),use_container_width=True)
        errors=[]
        for case in result.get("case_results",[]):
            for prediction in case["predictions"]:
                if (prediction["classification"]=="Contradiction") != prediction["expected_contradiction"]:
                    errors.append(dict(case=case["case_id"],top_k=case["top_k"],
                        statement_a=prediction["a"]["text"],statement_b=prediction["b"]["text"],
                        classification=prediction["classification"],expected_contradiction=prediction["expected_contradiction"],
                        probability=prediction["probabilities"]["contradiction"]))
        with st.expander("Misclassified document pairs"):
            st.dataframe(errors,use_container_width=True)
    st.download_button("Download evaluation JSON",__import__("json").dumps(result,indent=2),"evaluation.json")
    rows=result.get("examples",result.get("experiments",[]))
    st.download_button("Download evaluation CSV",pd.DataFrame(rows).to_csv(index=False),"evaluation.csv")

@st.fragment(run_every=2)
def job_panel(kind):
    jid=st.session_state.get(kind+"_job")
    if not jid:return
    job=data("GET",f"/{kind}/results/{jid}")
    if not job:return
    st.progress(float(job["progress"]),text=job["message"])
    if job["status"]=="failed":
        st.error(job["error"])
    elif job["status"]=="complete":
        st.session_state[kind+"_result"]=job["result"]
        if kind=="evaluation":show_evaluation(job["result"])
        else:st.success(f'Analysis complete: {job["result"]["candidate_pairs"]} candidate pairs. Use Refresh results below.')

def highlighted(text,other):
    tokens=text.split();comparison=other.split()
    matcher=difflib.SequenceMatcher(a=tokens,b=comparison)
    unchanged=set()
    for block in matcher.get_matching_blocks():unchanged.update(range(block.a,block.a+block.size))
    return " ".join(html.escape(token) if i in unchanged else "<mark>"+html.escape(token)+"</mark>" for i,token in enumerate(tokens))

def comparison_cards(rows):
    for i,row in enumerate(rows):
        with st.container(border=True):
            st.subheader(f'{row["classification"]} ? {row["probabilities"]["contradiction"]:.1%} contradiction probability')
            a,b=row["a"],row["b"]
            for col,left,right in ((st.columns(2),a,b),):
                for panel,claim,other in ((col[0],left,right),(col[1],right,left)):
                    panel.markdown(f'**{claim["filename"]}** ? Page {claim["page"] or "not available"}')
                    panel.markdown('<div class="claim">'+highlighted(claim["text"],other["text"])+"</div>",unsafe_allow_html=True)
                    with panel.expander("Supporting parent passage"):
                        st.write(claim["parent"])
            st.write(row["explanation"])
            with st.expander("Model probabilities and directional evidence"):
                st.json({k:row[k] for k in ("probabilities","forward","reverse","context_warnings")})

st.sidebar.markdown("## ?? DocuConflict AI")
st.sidebar.caption("LOCAL DOCUMENT INTELLIGENCE")
page=st.sidebar.radio("Workspace",["Dashboard","Upload Documents","Contradiction Finder","Ask About Documents","Evaluation Dashboard","Export Reports"])
st.sidebar.divider()
st.sidebar.caption("Offline inference ? Evidence first")
st.title(page)
st.caption("DocuConflict AI / Cross-document consistency workspace")
docs=data("GET","/documents") or []
lookup={d["id"]:d for d in docs}
result=st.session_state.get("analysis_result")
if page=="Dashboard":
    st.markdown("### Find where your documents disagree")
    st.write("Index policies, reports and specifications, then compare related claims with local natural language inference.")
    counts=Counter(r["classification"] for r in result["results"]) if result else Counter()
    values=[("Documents",len(docs)),("Statements",sum(d["statements"] for d in docs)),
        ("Candidate pairs",result["candidate_pairs"] if result else 0),("Contradictions",counts["Contradiction"]),
        ("Agreements",counts["Agreement"]),("Needs review",counts["Needs Review"])]
    columns=st.columns(3)
    for i,(name,value) in enumerate(values):columns[i%3].metric(name,value)
    if counts:
        st.plotly_chart(px.pie(names=list(counts),values=list(counts.values()),hole=.65),use_container_width=True)
    else:st.info("Upload two or more documents to begin. Dashboard statistics reflect the selected analysis.")
    history=data("GET","/analysis") or []
    complete=[j for j in history if j["status"]=="complete"]
    if complete:
        selected=st.selectbox("Load a saved analysis",[j["id"] for j in complete])
        if st.button("Load analysis"):
            job=data("GET",f"/analysis/results/{selected}")
            if job:st.session_state.update(analysis_job=selected,analysis_result=job["result"]);st.rerun()
elif page=="Upload Documents":
    uploads=st.file_uploader("Add PDF, DOCX or UTF-8 TXT documents",type=["pdf","docx","txt"],accept_multiple_files=True)
    st.caption("Text PDFs only; OCR is not included. Default maximum size: 20 MB per file.")
    if st.button("Upload and index",disabled=not uploads,type="primary"):
        with st.spinner("Extracting statements and indexing local embeddings..."):
            output=data("POST","/documents/upload",files=[("files",(f.name,f.getvalue(),f.type)) for f in uploads],timeout=600)
        if output:
            for item in output["files"]:
                if "error" in item:st.error(f'{item["filename"]}: {item["error"]}')
                else:st.success(f'{item["document"]["filename"]}: '+("already indexed" if item["duplicate"] else "indexed"))
            docs=data("GET","/documents") or []
    for doc in docs:
        with st.expander(f'{doc["filename"]} ? {doc["statements"]} statements'):
            st.text(doc["preview"])
            if st.button("Delete document",key=doc["id"]):
                if data("DELETE","/documents/"+doc["id"]):st.rerun()
    confirmed=st.checkbox("I want to remove all indexed documents")
    if st.button("Clear database",disabled=not confirmed):
        if data("DELETE","/documents") is not None:st.rerun()
    if st.button("Start analysis",disabled=len(docs)<2):
        job=data("POST","/analysis/compare",json={"document_ids":[d["id"] for d in docs]})
        if job:st.session_state["analysis_job"]=job["id"]
    job_panel("analysis")
elif page=="Contradiction Finder":
    selected=st.multiselect("Documents to compare",list(lookup),default=list(lookup)[:5],format_func=lambda i:lookup[i]["filename"])
    one,two,three=st.columns(3)
    threshold=one.slider("Contradiction threshold",.5,1.,.75,.01)
    k=two.slider("Retrieval top-K",1,20,5)
    both=three.checkbox("Verify both directions",True)
    st.caption("0.75 is an initial threshold. Scores are uncalibrated. Scope checks can require human review.")
    if st.button("Compare documents",disabled=len(selected)<2,type="primary"):
        job=data("POST","/analysis/compare",json=dict(document_ids=selected,threshold=threshold,top_k=k,bidirectional=both))
        if job:st.session_state["analysis_job"]=job["id"]
    job_panel("analysis")
    if st.button("Refresh results"):st.rerun()
    result=st.session_state.get("analysis_result")
    if result:
        classes=st.multiselect("Classifications",["Contradiction","Agreement","Neutral","Needs Review"],default=["Contradiction","Needs Review"])
        descending=st.checkbox("Highest contradiction probability first",True)
        rows=sorted([r for r in result["results"] if r["classification"] in classes],key=lambda r:r["probabilities"]["contradiction"],reverse=descending)
        st.caption(f"{len(rows)} matching pairs. Text highlighting marks token differences, not proven causes.")
        page_number=st.number_input("Results page (20 pairs per page)",1,max(1,(len(rows)+19)//20),1)
        comparison_cards(rows[(page_number-1)*20:page_number*20])
elif page=="Ask About Documents":
    question=st.text_input("Ask a question",placeholder="Which documents contain conflicting deadlines?")
    if st.button("Find supporting evidence",disabled=len(question)<3):
        with st.spinner("Retrieving evidence..."):answer=data("POST","/ask",json={"question":question})
        if answer:
            st.write(answer["answer"])
            for evidence in answer["evidence"]:
                st.markdown(f'**{evidence["filename"]}** ? Page {evidence["page"] or "N/A"} ? similarity {evidence["similarity"]:.2f}')
                st.info(evidence["text"])
            comparison_cards(answer.get("conflicts",[]))
elif page=="Evaluation Dashboard":
    dataset=st.selectbox("Benchmark",["multinli","snli","documents"])
    split=st.selectbox("Validation split",["validation_matched","validation_mismatched"]) if dataset=="multinli" else "validation"
    count=st.select_slider("Cached sentence-pair sample size",[300,600,900],value=300)
    st.caption("Download the selected official validation sample using scripts.download_datasets first. Document evaluation uses the separately labeled synthetic benchmark.")
    if st.button("Run evaluation",type="primary"):
        job=data("POST","/evaluation/run",json=dict(dataset=dataset,split=split,count=count))
        if job:st.session_state["evaluation_job"]=job["id"]
    job_panel("evaluation")
elif page=="Export Reports":
    history=data("GET","/analysis") or []
    ids=[j["id"] for j in history if j["status"]=="complete"]
    if ids:
        selected=st.selectbox("Completed analysis",ids)
        if st.button("Prepare downloads"):
            prepared={}
            for format in ("csv","json","pdf"):
                response=api("GET",f"/reports/{selected}?format={format}")
                if response is not None:
                    prepared[format]=(response.content,response.headers["content-type"])
            st.session_state["report_downloads"]=(selected,prepared)
        saved=st.session_state.get("report_downloads")
        if saved and saved[0]==selected:
            for format,(content,mime) in saved[1].items():
                st.download_button(f"Download {format.upper()}",content,f"docuconflict.{format}",mime=mime)
    else:st.info("Complete an analysis before exporting reports.")
