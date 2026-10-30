import hashlib
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import Response
from backend.core import storage
from backend.core.config import DATA, MAX_MB, NLI_ID, OFFLINE
from backend.schemas.models import CompareRequest, EvaluationRequest, AskRequest, DocumentInfo, JobInfo
from backend.services.document_loader import extract
from backend.services.preprocessing import statements

log=logging.getLogger(__name__)
executor=ThreadPoolExecutor(max_workers=1)
# One inference worker and one shared lock prevent deletion during analysis.
lock=threading.RLock()

@asynccontextmanager
async def lifespan(app):
    for job in storage.listing("jobs"):
        if job["status"] in ("queued","running"):
            job.update(status="failed",error="Server restarted before job completed. Please run again.")
            storage.save("jobs",job["id"],job)
    yield

app=FastAPI(title="DocuConflict AI",version="1.0.0",lifespan=lifespan)

def new_job(kind,work):
    jid=uuid.uuid4().hex
    job=dict(id=jid,kind=kind,status="queued",progress=0,message="Queued",result=None,error=None)
    storage.save("jobs",jid,job)
    def execute():
        def update(p,message):
            job.update(progress=p,message=message)
            storage.save("jobs",jid,job)
        try:
            with lock:
                job["status"]="running"
                update(0,"Starting")
                result=work(update)
            job.update(status="complete",progress=1,message="Complete",result=result)
        except Exception as exc:
            log.exception("Job failed")
            job.update(status="failed",error=str(exc),message="Failed")
        storage.save("jobs",jid,job)
    executor.submit(execute)
    return job

def require_documents(ids):
    ids=list(dict.fromkeys(ids))
    if len(ids)<2:
        raise HTTPException(422,"Select at least two different documents.")
    if any(not storage.get("documents",i) for i in ids):
        raise HTTPException(404,"One or more documents do not exist.")
    return ids

@app.get("/health")
def health():
    return dict(status="ok",offline=OFFLINE,nli_model=NLI_ID)

@app.get("/documents",response_model=list[DocumentInfo])
def documents():
    return storage.listing("documents")

@app.post("/documents/upload")
def upload(files: list[UploadFile]=File(...)):
    if len(files)>20:
        raise HTTPException(422,"Upload at most 20 files per request.")
    from backend.services import vector_store
    output=[]
    with lock:
        for file in files:
            filename=Path((file.filename or "document").replace("\\","/")).name[:180]
            content=file.file.read(MAX_MB*1024*1024+1)
            if len(content)>MAX_MB*1024*1024:
                output.append(dict(filename=filename,error=f"File exceeds {MAX_MB} MB limit.")); continue
            did=hashlib.sha256(content).hexdigest()
            existing=storage.get("documents",did)
            if existing:
                output.append(dict(document=existing,duplicate=True)); continue
            indexed=False
            try:
                pages=extract(content,filename)
                rows=statements(pages,did,filename)
                indexed=True
                vector_store.add(rows)
                (DATA/"uploads"/(did+Path(filename).suffix.lower())).write_bytes(content)
                doc=dict(id=did,filename=filename,filetype=Path(filename).suffix.lower(),
                    statements=len(rows),preview="\n".join(t for _,t in pages)[:5000])
                storage.save("documents",did,doc)
                output.append(dict(document=doc,duplicate=False))
            except Exception as exc:
                if indexed:
                    try:
                        vector_store.collection().delete(where={"document_id":did})
                    except Exception:
                        log.exception("Upload rollback failed")
                output.append(dict(filename=filename,error=str(exc)))
    return dict(files=output)

@app.delete("/documents/{document_id}")
def delete_document(document_id: str):
    from backend.services.vector_store import collection
    with lock:
        doc=storage.get("documents",document_id)
        if not doc:
            raise HTTPException(404,"Document not found.")
        collection().delete(where={"document_id":document_id})
        storage.delete("documents",document_id)
        (DATA/"uploads"/(document_id+doc["filetype"])).unlink(missing_ok=True)
    return dict(deleted=document_id)

@app.delete("/documents")
def clear_documents():
    with lock:
        for doc in storage.listing("documents"):
            delete_document(doc["id"])
    return dict(status="cleared",note="Historical reports are retained.")

@app.post("/analysis/compare",response_model=JobInfo)
def compare(request: CompareRequest):
    ids=require_documents(request.document_ids)
    def work(update):
        require_documents(ids)
        from backend.services.contradiction_engine import analyze
        return analyze(ids,request.threshold,request.top_k,request.bidirectional,update)
    return new_job("analysis",work)

@app.get("/analysis/results/{analysis_id}",response_model=JobInfo)
def analysis_result(analysis_id: str):
    return fetch_job(analysis_id,"analysis")

def fetch_job(jid,kind):
    job=storage.get("jobs",jid)
    if not job or job["kind"]!=kind:
        raise HTTPException(404,"Job not found.")
    return job

@app.get("/analysis")
def analysis_history():
    return [{k:v for k,v in j.items() if k!="result"} for j in storage.listing("jobs") if j["kind"]=="analysis"]

@app.post("/evaluation/run",response_model=JobInfo)
def run_evaluation(request: EvaluationRequest):
    if request.dataset=="multinli" and request.split=="validation":
        raise HTTPException(422,"MultiNLI requires validation_matched or validation_mismatched.")
    def work(update):
        from backend.services.evaluation import run
        return run(request,update)
    return new_job("evaluation",work)

@app.get("/evaluation/results/{evaluation_id}",response_model=JobInfo)
def evaluation_result(evaluation_id: str):
    return fetch_job(evaluation_id,"evaluation")

@app.get("/reports/{analysis_id}")
def report(analysis_id: str,format: str=Query("pdf",pattern="^(csv|json|pdf)$")):
    job=fetch_job(analysis_id,"analysis")
    if job["status"]!="complete":
        raise HTTPException(409,"Analysis is not complete.")
    from backend.services.report_generator import generate
    content,mime=generate(job["result"],format)
    return Response(content,media_type=mime,headers={"Content-Disposition":f'attachment; filename="docuconflict-{analysis_id}.{format}"'})

@app.post("/ask")
def ask(request: AskRequest):
    from backend.services.embeddings import encode
    from backend.services.vector_store import search
    with lock:
        ids=request.document_ids or [d["id"] for d in documents()]
        if not ids:
            return dict(answer="No supporting evidence found. Upload and index documents first.",evidence=[])
        evidence=search(encode([request.question])[0],ids,k=8)
        evidence=[r for r in evidence if r["similarity"]>=.25]
    if not evidence:
        return dict(answer="No sufficiently relevant supporting evidence found.",evidence=[])
    evidence_ids={r["id"] for r in evidence}
    conflicts=[]
    for job in storage.listing("jobs"):
        if job["kind"]=="analysis" and job["status"]=="complete":
            conflicts.extend(r for r in job["result"]["results"] if r["classification"]=="Contradiction"
                and r["a"]["document_id"] in ids and r["b"]["document_id"] in ids
                and (r["a"]["id"] in evidence_ids or r["b"]["id"] in evidence_ids))
    answer=(f"Found {len(conflicts)} relevant conflict records in saved analyses. " if conflicts else
        "No relevant confirmed conflicts in saved analyses. Run a comparison to test for contradictions. ")
    return dict(answer=answer+"Retrieved passages follow; this is an evidence summary, not a generated factual answer.",
        evidence=evidence,conflicts=conflicts[:10])
