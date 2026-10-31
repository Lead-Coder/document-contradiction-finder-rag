"""Run with the backend stopped; retains uploaded originals and old collections."""
import json
from backend.services.vector_store import rebuild
if __name__=="__main__":
    print(json.dumps(rebuild(lambda p,m:print(f"{p:.0%} {m}",flush=True)),indent=2))
