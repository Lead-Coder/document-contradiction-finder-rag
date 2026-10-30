import argparse
import json
import random
from backend.core.config import DATA

def prepare(dataset="multinli",split="validation_matched",count=300,seed=42):
    from datasets import load_dataset
    repo="nyu-mll/multi_nli" if dataset=="multinli" else "stanfordnlp/snli"
    if dataset=="snli":
        split="validation"
    # Streaming avoids downloading entire training datasets. Fixed-seed
    # buffer shuffle selects reproducible official-validation examples.
    stream=load_dataset(repo,split=split,streaming=True)
    names=stream.features["label"].names
    quotas={name:count//3+(i<count%3) for i,name in enumerate(names)}
    rows=[]
    for row in stream.shuffle(seed=seed,buffer_size=2000):
        label=row["label"]
        if label<0 or label>=len(names):
            continue
        name=names[label]
        if quotas.get(name,0)<=0 or not row["premise"].strip() or not row["hypothesis"].strip():
            continue
        rows.append(dict(premise=row["premise"],hypothesis=row["hypothesis"],label=label,label_name=name))
        quotas[name]-=1
        if all(n==0 for n in quotas.values()):
            break
    random.Random(seed).shuffle(rows)
    payload=dict(dataset=repo,split=split,seed=seed,label_names=names,requested=count,examples=rows)
    path=DATA/"datasets"/f"{dataset}_{split}_{count}.json"
    path.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"Saved {len(rows)} validation examples to {path}")
    return path

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",choices=["multinli","snli"],default="multinli")
    p.add_argument("--split",choices=["validation_matched","validation_mismatched","validation"],default="validation_matched")
    p.add_argument("--count",type=int,default=300)
    args=p.parse_args()
    if not 3<=args.count<=1000:
        p.error("count must be between 3 and 1000")
    if args.dataset=="multinli" and args.split=="validation":
        p.error("MultiNLI has validation_matched and validation_mismatched splits")
    prepare(args.dataset,args.split,args.count)
