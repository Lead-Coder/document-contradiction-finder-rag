import json
import sqlite3
from contextlib import contextmanager
from backend.core.config import DATA

@contextmanager
def connect():
    db=sqlite3.connect(DATA / "registry.sqlite3",timeout=30)
    db.row_factory=sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
    try:
        yield db
        db.commit()
    finally:
        db.close()

def save(table,key,value):
    assert table in ("documents","jobs")
    with connect() as db:
        db.execute(f"INSERT OR REPLACE INTO {table} VALUES (?,?)",(key,json.dumps(value)))

def get(table,key):
    assert table in ("documents","jobs")
    with connect() as db:
        row=db.execute(f"SELECT payload FROM {table} WHERE id=?",(key,)).fetchone()
    return json.loads(row[0]) if row else None

def listing(table):
    assert table in ("documents","jobs")
    with connect() as db:
        rows=db.execute(f"SELECT payload FROM {table}").fetchall()
    return [json.loads(row[0]) for row in rows]

def delete(table,key):
    assert table in ("documents","jobs")
    with connect() as db:
        db.execute(f"DELETE FROM {table} WHERE id=?",(key,))
