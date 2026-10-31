"""Frozen synthetic financial benchmark; labels are used only for scoring."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from backend.core.config import DATA
from backend.services.nli_classifier import predict, load

PATH=DATA/"benchmarks"/"financial_pairs.json"
LABELS=["Contradiction","Agreement","Neutral","Needs Review","Policy Change"]

def generate():
    if PATH.exists():return
    rows=[]
    for split,category,amount,other in [("development","standard","50,000","20,000"),("test","premium","80,000","30,000")]:
        claims=[
            (f"The daily transfer limit for {category} accounts is INR {amount}.",f"The daily transfer limit for {category} accounts is INR {other}."),
            (f"The daily withdrawal ceiling for {category} accounts is INR {amount}.",f"The daily withdrawal ceiling for {category} accounts is INR {other}."),
            (f"The monthly maintenance fee for {category} accounts is INR 100.",f"The monthly maintenance fee for {category} accounts is INR 200."),
            (f"Refunds for {category} accounts are completed in exactly five business days.",f"Refunds for {category} accounts are completed in exactly eight business days."),
            (f"The annual interest rate for {category} accounts is exactly 4 percent.",f"The annual interest rate for {category} accounts is exactly 6 percent."),
            (f"Minors are eligible to open {category} accounts.",f"Minors are not eligible to open {category} accounts."),
            (f"Identity verification is required for {category} account holders.",f"Identity verification is not required for {category} account holders."),
            (f"Complaints from {category} account holders are resolved in exactly seven calendar days.",f"Complaints from {category} account holders are resolved in exactly fourteen calendar days."),
            (f"International card payments are permitted for {category} accounts.",f"International card payments are prohibited for {category} accounts."),
        ]
        agreeing=[
            (f"Customers with {category} accounts may withdraw INR {amount} daily.",f"The daily cash withdrawal ceiling for {category} accounts is INR {amount}."),
            (f"Refunds for {category} accounts are completed within five working days.",f"Refunds for {category} accounts are completed within five business days."),
            (f"The annual maintenance fee for {category} accounts is INR 500.",f"The yearly account maintenance charge for {category} accounts is INR 500."),
            (f"The annual interest rate on {category} deposits is 5 percent.",f"The yearly interest rate on {category} deposits is 5%."),
            (f"Customers may close {category} accounts without paying a fee.",f"Closing a {category} account is free of charge."),
            (f"Customers must verify their identity to open {category} accounts.",f"Identity verification is mandatory when opening {category} accounts."),
        ]
        neutral=[
            (f"The daily transfer limit for {category} accounts is INR {amount}.","Customer support operates on weekdays."),
            (f"The annual maintenance fee for {category} accounts is INR 500.","Customers receive replacement cards by post."),
            ("Internal policy reference - Document 2 of 2 - Version 1.5 - Effective 1 October 2026","CedarBank is a fictional organization; no policy below describes a real financial institution."),
            ("Copyright 2026. All rights reserved.",f"The daily withdrawal ceiling for {category} accounts is INR {amount}."),
            (f"Refunds for {category} accounts are processed within five business days.",f"Refunds for {category} accounts are processed within ten business days."),
            (f"The bank provides statements for {category} accounts electronically.","The help desk has a wheelchair-accessible entrance."),
        ]
        contexts=[
            ("Basic account holders may withdraw INR 20,000 daily.","Premium account holders may withdraw INR 50,000 daily.","Needs Review",{},{}),
            ("The refund processing period was seven days in 2024.","The refund processing period is three days in 2026.","Policy Change",{},{}),
            (f"The daily transfer limit for {category} accounts is INR {amount}.",f"The daily transfer limit for {category} accounts is INR {other}.","Policy Change",{"version":"1.0","organization":"CedarBank"},{"version":"2.0","organization":"CedarBank"}),
            (f"The maintenance fee for {category} accounts is USD 10 monthly.",f"The maintenance fee for {category} accounts is INR 800 monthly.","Needs Review",{},{}),
        ]
        for label,examples in [("Contradiction",claims),("Agreement",agreeing),("Neutral",neutral)]:
            for a,b in examples:
                rows.append(dict(id=f"fin_{len(rows):03}",split=split,premise=a,hypothesis=b,expected=label,metadata_a={},metadata_b={}))
        for a,b,label,ma,mb in contexts:
            rows.append(dict(id=f"fin_{len(rows):03}",split=split,premise=a,hypothesis=b,expected=label,metadata_a=ma,metadata_b=mb))
    PATH.parent.mkdir(parents=True,exist_ok=True)
    PATH.write_text(json.dumps(dict(synthetic=True,seed=42,description="Manually specified labels; fixed development/test split. No training. Related template families appear in both splits, limiting independence.",examples=rows),indent=2),encoding="utf-8")

def metrics(rows):
    gold=[r["expected"]=="Contradiction" for r in rows]
    pred=[r["decision"]=="Contradiction" for r in rows]
    p,r,f,_=precision_recall_fscore_support(gold,pred,average="binary",zero_division=0)
    tp=sum(a and b for a,b in zip(gold,pred));fp=sum(not a and b for a,b in zip(gold,pred))
    tn=sum(not a and not b for a,b in zip(gold,pred));fn=sum(a and not b for a,b in zip(gold,pred))
    return dict(count=len(rows),precision=float(p),recall=float(r),f1=float(f),true_positives=tp,false_positives=fp,
        false_negatives=fn,false_positive_rate=fp/(fp+tn) if fp+tn else 0,
        labels=LABELS,confusion_matrix=confusion_matrix([r["expected"] for r in rows],[r["decision"] for r in rows],labels=LABELS).tolist())

def run(stage):
    generate()
    payload=json.loads(PATH.read_text(encoding="utf-8"));rows=payload["examples"]
    from backend.services import contradiction_engine as engine
    from backend.services.context_verification import verify
    if stage=="before" and hasattr(engine,"decide_pair"):
        raise ValueError("The original baseline was captured before the fixes. It cannot be recreated by labeling the updated detector as 'before'. Keep financial_before.json.")
    from backend.services.embeddings import encode
    _,model,mapping=load()
    print("id2label",model.config.id2label,flush=True);print("label2id",model.config.label2id,flush=True)
    inputs=[(r["premise"],r["hypothesis"]) for r in rows]
    probs=predict(inputs+[(b,a) for a,b in inputs])
    vectors=encode([r["premise"] for r in rows]+[r["hypothesis"] for r in rows])
    frozen=DATA/"results"/"financial_thresholds.json"
    settings=json.loads(frozen.read_text()) if stage=="after" and frozen.exists() else {"threshold":.75,"min_similarity":.35}
    output=[];started=time.perf_counter()
    for i,row in enumerate(rows):
        a=dict(id=row["id"]+"a",text=row["premise"],parent=row["premise"],document_id="financial_a",filename="financial_a.txt",page=0,**row["metadata_a"])
        b=dict(id=row["id"]+"b",text=row["hypothesis"],parent=row["hypothesis"],document_id="financial_b",filename="financial_b.txt",page=0,**row["metadata_b"])
        similarity=float(np.dot(vectors[i],vectors[i+len(rows)]))
        if hasattr(engine,"decide_pair"):
            result=engine.decide_pair(a,b,probs[i],probs[i+len(rows)],settings["threshold"],similarity=similarity,mode="strict",min_similarity=settings["min_similarity"])
            decision=result["classification"]
        else:
            decision,_=engine.classify(probs[i],probs[i+len(rows)],.75,verify(a,b))
            result={}
        output.append(dict(**row,decision=decision,similarity=similarity,forward=probs[i],reverse=probs[i+len(rows)],details=result))
    report=dict(stage=stage,synthetic=True,benchmark_sha256=hashlib.sha256(PATH.read_bytes()).hexdigest(),
        model_id2label=model.config.id2label,model_label2id=model.config.label2id,threshold=settings["threshold"],min_similarity=settings["min_similarity"],
        metrics={s:metrics([r for r in output if r["split"]==s]) for s in ("development","test")},
        examples=output,decision_seconds=time.perf_counter()-started)
    # Independent raw NLI probes, retained regardless of whether they match expectations.
    probes=[("The transaction limit is INR 50,000 per day.","The transaction limit is INR 20,000 per day."),
        ("The transaction limit is INR 50,000 per day.","The customer support team operates on weekdays."),
        ("Customers receive refunds within five business days.","Refunds are completed within five working days."),
        ("The transaction limit was INR 20,000 in 2024.","The transaction limit is INR 50,000 in 2026.")]
    report["independent_nli_probes"]=[dict(premise=a,hypothesis=b,probabilities=p) for (a,b),p in zip(probes,predict(probes))]
    path=DATA/"results"/f"financial_{stage}.json";path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report["metrics"],indent=2),flush=True)
    print("Saved",path,flush=True)
    return report

def tune():
    from backend.services.contradiction_engine import decide_pair
    generate()
    payload=json.loads(PATH.read_text(encoding="utf-8"))
    dev=[r for r in payload["examples"] if r["split"]=="development"]
    from backend.services.embeddings import encode
    inputs=[(r["premise"],r["hypothesis"]) for r in dev]
    scores=predict(inputs+[(b,a) for a,b in inputs])
    vectors=encode([r["premise"] for r in dev]+[r["hypothesis"] for r in dev])
    dev=[dict(r,forward=scores[i],reverse=scores[i+len(dev)],
        similarity=float(np.dot(vectors[i],vectors[i+len(dev)]))) for i,r in enumerate(dev)]
    fingerprint=hashlib.sha256(PATH.read_bytes()).hexdigest()
    candidates=[]
    for threshold in (.75,.85,.95):
        for minimum in (.35,.45,.55):
            predictions=[]
            for row in dev:
                a=dict(text=row["premise"],parent=row["premise"],**row["metadata_a"])
                b=dict(text=row["hypothesis"],parent=row["hypothesis"],**row["metadata_b"])
                result=decide_pair(a,b,row["forward"],row["reverse"],threshold,similarity=row["similarity"],min_similarity=minimum)
                predictions.append(dict(expected=row["expected"],decision=result["classification"]))
            measured=metrics(predictions)
            candidates.append(dict(threshold=threshold,min_similarity=minimum,**{k:measured[k] for k in ("precision","recall","f1","false_positive_rate")}))
    selected=max(candidates,key=lambda c:(c["precision"],c["f1"],c["recall"],c["threshold"],-c["min_similarity"]))
    result=dict(threshold=selected["threshold"],min_similarity=selected["min_similarity"],development_count=len(dev),
        candidates=candidates,benchmark_sha256=fingerprint,selection="Precision first, then F1/recall; ties prefer a higher NLI threshold and lower retrieval cutoff. Test labels not accessed.")
    (DATA/"results"/"financial_thresholds.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)
    return result

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=["before","after","tune"],required=True)
    stage=parser.parse_args().stage
    tune() if stage=="tune" else run(stage)
