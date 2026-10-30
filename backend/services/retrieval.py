from backend.services.vector_store import all_statements, search
from backend.core.config import MAX_PAIRS

def candidates(ids, top_k, progress=None):
    rows = all_statements(ids)
    found, seen = [], set()
    for i,a in enumerate(rows):
        for b in search(a["embedding"], ids, top_k, exclude=a["document_id"]):
            key = tuple(sorted((a["id"],b["id"])))
            if key in seen:
                continue
            seen.add(key)
            clean_a = {k:v for k,v in a.items() if k != "embedding"}
            found.append((clean_a,b))
            if len(found) > MAX_PAIRS:
                raise ValueError(f"More than {MAX_PAIRS} candidate pairs; reduce documents or top-K.")
        if progress:
            progress((i+1)/len(rows))
    return found
