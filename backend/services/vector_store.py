from functools import lru_cache
from backend.core.config import DATA, EMBED_ID
from backend.services.embeddings import encode

def client():
    import chromadb
    from chromadb.config import Settings
    return chromadb.PersistentClient(path=str(DATA / "chroma_db"),settings=Settings(anonymized_telemetry=False))


def index_status():
    import json
    from backend.services.preprocessing import INDEX_SCHEMA
    path=DATA/"index_manifest.json"
    manifest=json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    names=[c.name for c in client().list_collections()]
    legacy_count=client().get_collection("statements",embedding_function=None).count() if "statements" in names else 0
    current=manifest.get("schema")==INDEX_SCHEMA and manifest.get("collection") in names
    return dict(schema=INDEX_SCHEMA,ready=current or not legacy_count,requires_rebuild=not current and bool(legacy_count),
        active_collection=manifest.get("collection"),legacy_claims=legacy_count,collections=names)


def write_manifest(name):
    import json
    from backend.services.preprocessing import INDEX_SCHEMA
    temporary=DATA/"index_manifest.tmp"
    temporary.write_text(json.dumps(dict(collection=name,schema=INDEX_SCHEMA,embedding_model=EMBED_ID)),encoding="utf-8")
    temporary.replace(DATA/"index_manifest.json")


@lru_cache(maxsize=1)
def collection():
    import json
    from backend.services.preprocessing import INDEX_SCHEMA
    status=index_status()
    if status["requires_rebuild"]:
        raise ValueError("Stored claims use an old preprocessing schema. Rebuild the index from Upload Documents or python -m scripts.rebuild_index. Source files are retained.")
    name=status["active_collection"] or "statements_v2"
    col=client().get_or_create_collection(name,embedding_function=None,
        metadata={"embedding_model":EMBED_ID,"index_schema":INDEX_SCHEMA},configuration={"hnsw":{"space":"cosine"}})
    if col.metadata.get("embedding_model")!=EMBED_ID or col.metadata.get("index_schema")!=INDEX_SCHEMA:
        raise ValueError("Index configuration differs; rebuild before querying.")
    if not status["active_collection"]:write_manifest(name)
    return col


def add(rows,target=None):
    if not rows:return
    vectors=encode([r["text"] for r in rows])
    col=target if target is not None else collection()
    for start in range(0,len(rows),128):
        part=rows[start:start+128]
        col.upsert(ids=[r["id"] for r in part],documents=[r["text"] for r in part],embeddings=vectors[start:start+128],
            metadatas=[{k:v for k,v in r.items() if k not in ("id","text") and isinstance(v,(str,int,float,bool))} for r in part])


def rebuild(progress=lambda p,m:None):
    import hashlib,json,uuid
    from backend.core import storage
    from backend.services.document_loader import extract
    from backend.services.preprocessing import statements,INDEX_SCHEMA,document_metadata
    docs=storage.listing("documents")
    name="statements_v2_"+uuid.uuid4().hex
    col=client().create_collection(name,embedding_function=None,metadata={"embedding_model":EMBED_ID,"index_schema":INDEX_SCHEMA},
        configuration={"hnsw":{"space":"cosine"}})
    updated=[];expected_ids=set()
    try:
        for i,doc in enumerate(docs):
            source=(DATA/"uploads"/(doc["id"]+doc["filetype"])).resolve()
            if source.parent!=(DATA/"uploads").resolve():raise ValueError("Invalid stored source path")
            content=source.read_bytes()
            if hashlib.sha256(content).hexdigest()!=doc["id"]:raise ValueError("Stored source hash mismatch: "+doc["filename"])
            pages=extract(content,doc["filename"])
            try:rows=statements(pages,doc["id"],doc["filename"])
            except ValueError as exc:
                if "No meaningful policy claims" not in str(exc):raise
                rows=[]
            add(rows,target=col)
            expected_ids.update(r["id"] for r in rows)
            updated.append(dict(doc,statements=len(rows),index_schema=INDEX_SCHEMA,metadata=document_metadata(pages)))
            progress((i+1)/max(1,len(docs)),"Reindexing "+doc["filename"])
        found=col.get(include=["metadatas","documents"])
        if set(found["ids"])!=expected_ids or col.count()!=len(expected_ids):raise ValueError("Rebuild claim-ID validation failed")
        if any(m["document_id"] not in {d["id"] for d in docs} for m in found["metadatas"]):raise ValueError("Invalid document association")
        backup=DATA/("registry_documents_backup_"+uuid.uuid4().hex+".json")
        backup.write_text(json.dumps(docs,indent=2),encoding="utf-8")
        with storage.connect() as db:
            db.executemany("INSERT OR REPLACE INTO documents VALUES (?,?)",[(d["id"],json.dumps(d)) for d in updated])
        write_manifest(name)
        collection.cache_clear()
        return dict(schema=INDEX_SCHEMA,collection=name,claims=col.count(),documents=len(docs),
            registry_backup=backup.name,source_files_preserved=True,old_collections_preserved=True)
    except Exception:
        # Never activate a partially built index. The failed collection is retained for inspection.
        collection.cache_clear()
        raise

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
