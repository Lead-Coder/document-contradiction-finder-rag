from functools import lru_cache
from backend.core.config import DATA, EMBED_ID
from backend.services.embeddings import encode

@lru_cache(maxsize=1)
def collection():
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(path=str(DATA / "chroma_db"),
        settings=Settings(anonymized_telemetry=False))
    col = client.get_or_create_collection("statements", embedding_function=None,
        metadata={"embedding_model": EMBED_ID}, configuration={"hnsw": {"space": "cosine"}})
    if col.metadata.get("embedding_model") != EMBED_ID:
        raise ValueError("Embedding model changed; clear and re-index the database.")
    return col

def add(rows):
    vectors = encode([r["text"] for r in rows])
    for start in range(0,len(rows),128):
        part = rows[start:start+128]
        collection().upsert(ids=[r["id"] for r in part], documents=[r["text"] for r in part],
            embeddings=vectors[start:start+128], metadatas=[{k:v for k,v in r.items() if k not in ("id","text")} for r in part])

def all_statements(ids):
    data = collection().get(where={"document_id": {"$in": ids}}, include=["documents", "metadatas", "embeddings"])
    return [dict(id=sid, text=text, **meta, embedding=vec.tolist()) for sid,text,meta,vec in
        zip(data["ids"],data["documents"],data["metadatas"],data["embeddings"])]

def search(vector, ids, k=5, exclude=None):
    if not ids:
        return []
    where = {"document_id": {"$in": ids}}
    if exclude:
        where = {"$and": [where, {"document_id": {"$ne": exclude}}]}
    count = len(collection().get(where=where, include=[])["ids"])
    if not count:
        return []
    found = collection().query(query_embeddings=[vector], where=where,
        n_results=min(k,count), include=["documents","metadatas","distances"])
    return [dict(id=sid,text=text,**meta,similarity=1-float(distance)) for sid,text,meta,distance in
        zip(found["ids"][0],found["documents"][0],found["metadatas"][0],found["distances"][0])]
