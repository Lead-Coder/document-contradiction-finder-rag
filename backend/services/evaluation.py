import csv
import hashlib
import json
import time
import uuid
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from backend.core.config import DATA
from backend.services.nli_classifier import predict

LABELS=["contradiction","entailment","neutral"]

def pair_key(a,b):
    # IDs preserve distinct documents, including duplicated claim text.
    return tuple(sorted((a,b)))

def save_result(result):
    import platform
    from datetime import datetime,timezone
    from backend.core.config import NLI_ID,EMBED_ID,DEVICE
    from backend.services.nli_classifier import load
    result.update(model=NLI_ID,embedding_model=EMBED_ID,
        model_revision=getattr(load()[1].config,"_commit_hash",None),
        timestamp=datetime.now(timezone.utc).isoformat(),
        environment=dict(python=platform.python_version(),system=platform.system(),
            processor=platform.processor(),device=DEVICE))

    stem=DATA/"results"/f"evaluation_{uuid.uuid4().hex}"
    result["artifact_stem"]=stem.name
    stem.with_suffix(".json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    rows=result.get("examples",result.get("experiments",[]))
    if rows:
        with stem.with_suffix(".csv").open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import plotly.express as px
    if "confusion_matrix" in result:
        fig=px.imshow(result["confusion_matrix"],x=LABELS,y=LABELS,text_auto=True,
            labels=dict(x="Predicted",y="Actual"),title="NLI confusion matrix")
    else:
        fig=px.bar(result["experiments"],x="top_k",y=["recall_at_k","precision","recall","f1"],barmode="group")
    fig.write_html(str(stem.with_suffix(".html")),include_plotlyjs=True)
    return result

def sentence_evaluation(request,update):
    split="validation" if request.dataset=="snli" else request.split
    path=DATA/"datasets"/f"{request.dataset}_{split}_{request.count}.json"
    if not path.exists():
        raise ValueError(f"Evaluation cache missing. Run python -m scripts.download_datasets --dataset {request.dataset} --split {split} --count {request.count} while online.")
    payload=json.loads(path.read_text(encoding="utf-8"))
    rows=payload["examples"]
    if not rows:
        raise ValueError("Dataset cache contains no valid examples.")
    start=time.perf_counter()
    probabilities=predict([(r["premise"],r["hypothesis"]) for r in rows],lambda p:update(p,"Classifying validation examples"))
    gold=[r["label_name"] for r in rows]
    predicted=[max(p,key=p.get) for p in probabilities]
    metrics=classification_report(gold,predicted,labels=LABELS,output_dict=True,zero_division=0)
    return save_result(dict(kind="sentence_nli",dataset=payload["dataset"],split=split,seed=payload["seed"],
        count=len(rows),accuracy=accuracy_score(gold,predicted),macro_precision=metrics["macro avg"]["precision"],
        macro_recall=metrics["macro avg"]["recall"],macro_f1=metrics["macro avg"]["f1-score"],
        per_class=metrics,confusion_matrix=confusion_matrix(gold,predicted,labels=LABELS).tolist(),
        labels=LABELS,runtime_seconds=time.perf_counter()-start,
        examples=[dict(**r,prediction=pred,probabilities=p,correct=r["label_name"]==pred)
            for r,pred,p in zip(rows,predicted,probabilities)]))

def document_evaluation(request,update):
    import chromadb
    from backend.services.document_loader import extract
    from backend.services.preprocessing import statements
    from backend.services.embeddings import encode
    from backend.services.context_verification import verify
    from backend.services.contradiction_engine import classify
    root=DATA/"benchmarks"
    path=root/"ground_truth.json"
    if not path.exists():
        raise ValueError("Generate benchmark first: python -m scripts.generate_sample_documents")
    cases=json.loads(path.read_text(encoding="utf-8"))
    # Isolated ephemeral Chroma collections never change user-uploaded documents.
    from chromadb.config import Settings
    client=chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    prepared=[]
    for case in cases:
        rows=[]
        for i,doc in enumerate(case["documents"]):
            did=f'{case["id"]}_{i}'
            rows+=statements(extract((root/doc["filename"]).read_bytes(),doc["filename"]),did,doc["filename"])
        vectors=encode([r["text"] for r in rows])
        prepared.append((case,rows,vectors))
    experiments=[]
    diagnostics=[]
    for ki,k in enumerate((3,5,10)):
        start=time.perf_counter();tp=fp=fn=retrieved=total_gold=pair_count=positive_cases=successful_cases=0
        for ci,(case,rows,vectors) in enumerate(prepared):
            col=client.create_collection("eval_"+uuid.uuid4().hex,embedding_function=None,
                configuration={"hnsw":{"space":"cosine"}})
            try:
                col.add(ids=[r["id"] for r in rows],documents=[r["text"] for r in rows],embeddings=vectors,
                    metadatas=[{"document_id":r["document_id"]} for r in rows])
                by_id={r["id"]:r for r in rows}
                candidates=set()
                for row,vector in zip(rows,vectors):
                    count=sum(r["document_id"]!=row["document_id"] for r in rows)
                    found=col.query(query_embeddings=[vector],n_results=min(k,count),
                        where={"document_id":{"$ne":row["document_id"]}},include=[])
                    candidates.update(pair_key(row["id"],sid) for sid in found["ids"][0])
                # Gold is used only AFTER retrieval, for scoring; never as detector input.
                gold=set()
                for a,b in case["expected_conflicts"]:
                    gold.update(pair_key(x["id"],y["id"]) for x in rows for y in rows
                        if x["text"]==a and y["text"]==b and x["document_id"]!=y["document_id"])
                if case["expected_conflicts"] and not gold:
                    raise ValueError(f'Benchmark annotation did not align with extraction: {case["id"]}')
                keys=sorted(candidates)
                inputs=[(by_id[a]["text"],by_id[b]["text"]) for a,b in keys]
                probs=predict(inputs+[(b,a) for a,b in inputs]) if inputs else []
                detected=set()
                case_predictions=[]
                for j,(a,b) in enumerate(keys):
                    reasons=verify(by_id[a],by_id[b])
                    label,probabilities=classify(probs[j],probs[j+len(keys)],request.threshold,reasons)
                    case_predictions.append(dict(a=by_id[a],b=by_id[b],classification=label,
                        probabilities=probabilities,context_warnings=reasons,expected_contradiction=(a,b) in gold))
                    if label=="Contradiction":detected.add((a,b))
                diagnostics.append(dict(case_id=case["id"],top_k=k,predictions=case_predictions,
                    false_positive_pairs=[list(pair) for pair in sorted(detected-gold)],
                    false_negative_pairs=[list(pair) for pair in sorted(gold-detected)]))
                tp+=len(detected & gold);fp+=len(detected-gold);fn+=len(gold-detected)
                retrieved+=len(candidates & gold);total_gold+=len(gold);pair_count+=len(candidates)
                if gold:
                    positive_cases+=1;successful_cases+=bool(candidates & gold)
            finally:
                client.delete_collection(col.name)
            update((ki*len(cases)+ci+1)/(3*len(cases)),f"Benchmark top-K={k}: {case['id']}")
        precision=tp/(tp+fp) if tp+fp else 0
        recall=tp/(tp+fn) if tp+fn else 0
        experiments.append(dict(top_k=k,recall_at_k=retrieved/total_gold if total_gold else 0,
            candidate_success_rate=successful_cases/positive_cases if positive_cases else 0,
            candidate_pairs=pair_count,true_positives=tp,false_positives=fp,false_negatives=fn,
            precision=precision,recall=recall,f1=2*precision*recall/(precision+recall) if precision+recall else 0,
            runtime_seconds=time.perf_counter()-start))
    return save_result(dict(kind="synthetic_document_benchmark",synthetic=True,cases=len(cases),
        threshold=request.threshold,experiments=experiments,case_results=diagnostics,
        success_definition="Fraction of positive cases with at least one gold pair retrieved. Recall@K is pooled unique gold-pair coverage from bidirectional retrieval."))

def run(request,update=lambda p,m:None):
    return document_evaluation(request,update) if request.dataset=="documents" else sentence_evaluation(request,update)
