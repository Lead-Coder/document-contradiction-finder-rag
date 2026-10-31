import argparse
import json
from backend.schemas.models import EvaluationRequest
from backend.services.evaluation import run
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",choices=["multinli","snli","documents","financial"],default="multinli")
    p.add_argument("--split",default="validation_matched")
    p.add_argument("--count",type=int,default=300)
    p.add_argument("--threshold",type=float,default=.75)
    args=p.parse_args()
    result=run(EvaluationRequest(**vars(args)),lambda progress,message:print(f"{progress:.0%} {message}"))
    print(json.dumps({k:v for k,v in result.items() if k not in ("examples","case_results","end_to_end")},indent=2))
