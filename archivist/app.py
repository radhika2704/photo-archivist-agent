import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import db, embed, search as S

app = FastAPI(title="Photo Archivist")


@app.on_event("startup")
def warmup():
    threading.Thread(target=embed.available, daemon=True).start()  # load CLIP in background


@app.get("/api/search")
def api_search(q: str = Query(..., min_length=1), limit: int = 12):
    return S.search(q, limit)


def _row(file_id):
    row = db.connect().execute("SELECT * FROM files WHERE file_id=?", (file_id,)).fetchone()
    if not row:
        raise HTTPException(404, "unknown file")
    return row


@app.get("/api/file/{file_id}")
def api_file(file_id: str):
    row = _row(file_id)
    p = Path(row["organised_path"] or row["source_path"])
    if not p.exists():
        raise HTTPException(404, "file missing on disk")
    return FileResponse(p)


@app.get("/api/meta/{file_id}")
def api_meta(file_id: str):
    return db.row_to_dict(_row(file_id), with_raw=True)


@app.get("/api/stats")
def api_stats():
    c = db.connect()
    q = lambda s: c.execute(s).fetchone()[0]
    return {"files": q("SELECT COUNT(*) FROM files"),
            "images": q("SELECT COUNT(*) FROM files WHERE type='image'"),
            "documents": q("SELECT COUNT(*) FROM files WHERE type='document'"),
            "stripped": q("SELECT COUNT(*) FROM files WHERE metadata_status='stripped'"),
            "organised": q("SELECT COUNT(*) FROM files WHERE organised_path IS NOT NULL"),
            "duplicates": q("SELECT COUNT(*) FROM duplicates")}


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
