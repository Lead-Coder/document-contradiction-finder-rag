from backend.services.vector_store import all_statements, search
from backend.core.config import MAX_PAIRS

def candidates(ids, top_k, progress=None, min_similarity=.35, diagnostics=None, semantic=True):
    rows = all_statements(ids)
    found, seen = [], set()
    from backend.services.context_verification import candidate_check
    diagnostics=diagnostics if diagnostics is not None else {}
    diagnostics.update(retrieved_pairs=0,rejected_pairs=0,rejections=[])
    for i,a in enumerate(rows):
        for b in search(a["embedding"], ids, top_k, exclude=a["document_id"]):
            key = tuple(sorted((a["id"],b["id"])))
            if key in seen:
                continue
            seen.add(key)
            diagnostics["retrieved_pairs"]+=1
            keep,reason,fa,fb=candidate_check(a,b,b["similarity"],min_similarity,semantic=semantic)
            if not keep:
                diagnostics["rejected_pairs"]+=1
                diagnostics["rejections"].append(dict(a_id=a["id"],b_id=b["id"],similarity=b["similarity"],reason=reason,
                    a_text=a["text"],b_text=b["text"],a_document_id=a["document_id"],b_document_id=b["document_id"],
                    a_topic=fa["topic"],b_topic=fb["topic"],nli_evaluated=False))
                continue
            a=dict(a,topic=fa["topic"])
            b=dict(b,topic=fb["topic"])
            clean_a = {k:v for k,v in a.items() if k != "embedding"}
            found.append((clean_a,b))
            if len(found) > MAX_PAIRS:
                raise ValueError(f"More than {MAX_PAIRS} candidate pairs; reduce documents or top-K.")
        if progress:
            progress((i+1)/len(rows))
    return found
