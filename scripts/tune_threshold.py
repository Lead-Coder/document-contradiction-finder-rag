import argparse
import json
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support
from backend.core.config import DATA
from backend.services.nli_classifier import predict
from backend.services.context_verification import verify
from backend.services.contradiction_engine import classify

def main():
    parser=argparse.ArgumentParser(description="Tune on development-only data, never the evaluation benchmark")
    parser.add_argument("--development",type=Path,default=DATA/"development"/"pairs.json")
    args=parser.parse_args()
    payload=json.loads(args.development.read_text(encoding="utf-8"))
    if "Development" not in payload.get("purpose",""):
        raise ValueError("Input must explicitly declare a Development purpose")
    rows=payload["examples"]
    pairs=[(r["premise"],r["hypothesis"]) for r in rows]
    probs=predict(pairs+[(b,a) for a,b in pairs])
    gold=[r["contradiction"] for r in rows]
    scores=[]
    for threshold in (.5,.55,.6,.65,.7,.75,.8,.85,.9,.95):
        predicted=[]
        for i,(a,b) in enumerate(pairs):
            label,_=classify(probs[i],probs[i+len(pairs)],threshold,verify(dict(text=a),dict(text=b)))
            predicted.append(label=="Contradiction")
        p,r,f,_=precision_recall_fscore_support(gold,predicted,average="binary",zero_division=0)
        scores.append(dict(threshold=threshold,precision=float(p),recall=float(r),f1=float(f)))
    best=max(scores,key=lambda r:(r["f1"],r["precision"],r["threshold"]))
    result=dict(development_file=str(args.development),count=len(rows),selected=best,candidates=scores,
        warning="Small synthetic development set; freeze threshold before final evaluation.")
    out=DATA/"results"/"development_threshold.json"
    out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
    print("Set the interface slider or run_evaluation --threshold to the selected value; it is not applied silently.")

if __name__=="__main__":main()
