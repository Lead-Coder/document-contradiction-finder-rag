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

def analyze(ids,threshold=.75,top_k=5,bidirectional=True,progress=None):
    update=progress or (lambda p,m:None)
    update(.02,"Retrieving cross-document candidates")
    pairs=candidates(ids,top_k,lambda p:update(.02+.18*p,"Retrieving candidates"))
    inputs=[(a["text"],b["text"]) for a,b in pairs]
    if bidirectional:
        inputs += [(b["text"],a["text"]) for a,b in pairs]
    scores=predict(inputs,lambda p:update(.2+.7*p,"Running batched NLI")) if inputs else []
    results=[]
    for i,(a,b) in enumerate(pairs):
        forward=scores[i]
        reverse=scores[i+len(pairs)] if bidirectional else forward
        reasons=verify(a,b)
        label,prob=classify(forward,reverse,threshold,reasons)
        results.append(dict(a=a,b=b,classification=label,probabilities=prob,
            forward=forward,reverse=reverse if bidirectional else None,
            context_warnings=reasons,explanation=explain(a,b,label,prob,reasons)))
    results.sort(key=lambda r:r["probabilities"]["contradiction"],reverse=True)
    update(1,"Complete")
    from backend.core import storage
    names=[d["filename"] for did in ids if (d:=storage.get("documents",did))]
    return dict(document_names=names,timestamp=datetime.now(timezone.utc).isoformat(),documents=ids,
        model=NLI_ID,embedding_model=EMBED_ID,threshold=threshold,top_k=top_k,
        bidirectional=bidirectional,candidate_pairs=len(pairs),results=results,
        caveat="Probabilities are uncalibrated model estimates. Context checks are conservative heuristics.")
