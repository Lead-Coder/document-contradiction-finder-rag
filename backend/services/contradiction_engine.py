from datetime import datetime, timezone
from backend.services.retrieval import candidates
from backend.services.nli_classifier import predict
from backend.services.context_verification import verify
from backend.services.explanation import explain
from backend.core.config import NLI_ID, EMBED_ID

def classify(forward,reverse,threshold,reasons):
    scores={k:(forward[k]+reverse[k])/2 for k in forward}
    winners={max(p,key=p.get) for p in (forward,reverse)}
    if reasons or len(winners)>1:
        return "Needs Review",scores
    winner=max(scores,key=scores.get)
    if scores[winner] < (threshold if winner=="contradiction" else .6):
        return "Needs Review",scores
    return {"contradiction":"Contradiction","entailment":"Agreement","neutral":"Neutral"}[winner],scores


def decide_pair(a,b,forward=None,reverse=None,threshold=.75,similarity=1.0,mode="strict",min_similarity=.35):
    from backend.services.context_verification import candidate_check,scope,features
    keep,reason,fa,fb=candidate_check(a,b,similarity,min_similarity)
    checked=scope(a,b,fa,fb)
    warnings=list(checked["warnings"])
    # Retain generic entity/date safeguards for non-financial documents.
    if fa["topic"]==fb["topic"]=="unknown":
        warnings=list(dict.fromkeys(warnings+verify(a,b)))
    scores={k:(forward[k]+(reverse or forward)[k])/2 for k in forward} if forward else None
    if not keep:label="Neutral";warnings=[reason]+warnings
    elif warnings:label="Needs Review"
    elif forward:
        label,scores=classify(forward,reverse or forward,threshold,[])
        if checked["policy_changes"]:
            amounts=lambda f:{(q["value"],q["unit"],q["currency"]) for q in f["quantities"]}
            changed=label=="Contradiction" or amounts(fa)!=amounts(fb)
            if changed:label="Policy Change"
            elif label!="Agreement":label="Needs Review"
    else:label="Needs Review"
    context=dict(compatible=keep and not warnings and not checked["policy_changes"],warnings=warnings,
        policy_changes=checked["policy_changes"],candidate_reason=reason)
    result=dict(a=a,b=b,classification=label,probabilities=scores,forward=forward,reverse=reverse,
        nli_evaluated=forward is not None,topic=fa["topic"] if fa["topic"]==fb["topic"] else fa["topic"]+" / "+fb["topic"],
        retrieval_similarity=similarity,context_warnings=warnings+checked["policy_changes"],context_verification=context,
        mode=mode,features_a=fa,features_b=fb,candidate_accepted=keep)
    result["explanation"]=explain(a,b,label,scores,result["context_warnings"],fa,fb)
    return result


def analyze(ids,threshold=.75,top_k=5,bidirectional=True,progress=None,min_similarity=.35,mode="strict",debug=False):
    import logging,time
    from backend.services.preprocessing import INDEX_SCHEMA
    update=progress or (lambda p,m:None)
    started=time.perf_counter();diagnostics={}
    update(.02,"Retrieving and validating cross-document topics")
    pairs=candidates(ids,top_k,lambda p:update(.02+.18*p,"Filtering candidates"),min_similarity,diagnostics)
    prepared=[decide_pair(a,b,similarity=b["similarity"],mode=mode,min_similarity=min_similarity) for a,b in pairs]
    eligible=[i for i,r in enumerate(prepared) if not r["context_verification"]["warnings"]]
    inputs=[(pairs[i][0]["text"],pairs[i][1]["text"]) for i in eligible]
    if bidirectional:inputs += [(b,a) for a,b in inputs]
    scores=predict(inputs,lambda p:update(.2+.7*p,"Running batched NLI on relevant claims")) if inputs else []
    for j,i in enumerate(eligible):
        a,b=pairs[i];forward=scores[j];reverse=scores[j+len(eligible)] if bidirectional else None
        prepared[i]=decide_pair(a,b,forward,reverse,threshold,b["similarity"],mode,min_similarity)
    order={"Contradiction":0,"Policy Change":1,"Needs Review":2,"Agreement":3,"Neutral":4}
    prepared.sort(key=lambda r:(order[r["classification"]],-(r["probabilities"] or {}).get("contradiction",0)))
    if debug:
        for rejected in diagnostics["rejections"]:logging.getLogger(__name__).info("rejected_pair %s",__import__("json").dumps(rejected,ensure_ascii=True))
        for result in prepared:logging.getLogger(__name__).info("pair_diagnostic %s",__import__("json").dumps(result,ensure_ascii=True))
    from backend.core import storage
    names=[d["filename"] for did in ids if (d:=storage.get("documents",did))]
    update(1,"Complete")
    return dict(document_names=names,timestamp=datetime.now(timezone.utc).isoformat(),documents=ids,
        model=NLI_ID,embedding_model=EMBED_ID,threshold=threshold,top_k=top_k,min_similarity=min_similarity,mode=mode,
        index_schema=INDEX_SCHEMA,analysis_version="financial-v2",bidirectional=bidirectional,
        candidate_pairs=len(pairs),retrieved_pairs=diagnostics["retrieved_pairs"],rejected_pairs=diagnostics["rejected_pairs"],
        nli_pairs=len(eligible),candidate_rejection_rate=diagnostics["rejected_pairs"]/max(1,diagnostics["retrieved_pairs"]),
        diagnostics=diagnostics if debug else None,runtime_seconds=time.perf_counter()-started,results=prepared,
        caveat="NLI scores are uncalibrated model signals, not confirmed conflict probabilities. Policy changes are not simultaneous contradictions.")
