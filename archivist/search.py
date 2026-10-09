"""Hybrid search (spec section 6): structured filters (exact, cheap) -> semantic vectors + keyword hits -> rank,
every hit carries a reason + provenance."""
import calendar
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from . import config, db, embed

STOP = set("the a an of at in on for to from with and me give show find get photo photos image images picture "
           "pictures pic pics file files please i want need where is was were".split())
DOC_WORDS = {"letter", "letters", "document", "documents", "pdf", "sanction", "sanctions", "report", "agreement"}
IMG_WORDS = {"photo", "photos", "image", "images", "picture", "pictures", "pic", "pics"}
MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})

SOURCE_LABEL = {
    "exif_datetimeoriginal": "EXIF date", "gps_timestamp": "GPS timestamp", "takeout_json_phototakentime": "Takeout JSON",
    "quicktime_createdate": "QuickTime date", "xmp_createdate": "XMP date", "filename_embedded_date": "filename date",
    "doc_creation_date": "document properties", "filesystem_birthtime": "file birthtime (unreliable)",
    "filesystem_mtime": "file mtime (unreliable)",
    "metadata_region": "region tag", "takeout_json_people": "Takeout people",
    "exif_gps": "EXIF GPS", "exif_gps+reverse_geo": "EXIF GPS", "exif_gps+iptc": "EXIF GPS + IPTC",
    "iptc_xmp_location": "IPTC/XMP location", "hierarchical_keyword": "keywords", "source_path": "folder path",
    "vision_model_guess": "vision guess", "embedded_caption": "embedded caption", "vision_model": "vision model", "filename_hint": "filename hint (unverified)", "folder_hint": "folder name (unverified)", "folder_owner_hint": "folder name (unverified)",
}

SYN = {"suit": ["suit", "blazer", "jacket", "coat"], "blazer": ["blazer", "suit", "jacket"], "glasses": ["glasses", "spectacles", "eyeglasses"],
       "speaking": ["speaking", "speech", "podium", "talking", "presenting"], "brown": ["brown", "tan", "beige"]}
_cache = {"key": None}


def _roles():
    try:
        return {k.lower(): v for k, v in json.loads(config.ROLES_FILE.read_text()).items()}
    except Exception:
        return {}


def _load(conn):
    key = conn.execute("SELECT COUNT(*), MAX(indexed_at) FROM files").fetchone()
    key = tuple(key)
    if _cache["key"] == key:
        return _cache
    rows = [db.row_to_dict(r) for r in conn.execute("SELECT * FROM files")]
    vecs = {r["file_id"]: (embed.from_blob(r["img_vec"]), embed.from_blob(r["txt_vec"]))
            for r in conn.execute("SELECT file_id,img_vec,txt_vec FROM files")}
    _cache.update(key=key, rows=rows, vecs=vecs)
    return _cache


def parse_query(q: str, rows):
    ql = q.lower()
    f = {"years": [], "months": [], "type": None, "people": [], "places": [], "explain": []}
    f["years"] = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", ql)]
    for w in re.findall(r"[a-z]+", ql):
        if w in MONTHS:
            f["months"].append(MONTHS[w])
    words = set(re.findall(r"[a-z]+", ql))
    if words & DOC_WORDS:
        f["type"] = "document"
    elif words & IMG_WORDS:
        f["type"] = "image"
    roles = _roles()
    for role, name in roles.items():  # "the MD" -> named individual
        if re.search(rf"\b{re.escape(role)}\b", ql):
            f["people"].append(name)
            f["explain"].append(f"'{role.upper()}' resolved to {name}")
    known_people = {p["name"] for r in rows for p in (r["people"] or [])}
    for n in known_people:
        if n.lower() in ql or any(len(t) > 2 and re.search(rf"\b{re.escape(t)}\b", ql) for t in n.lower().split()[:1]):
            if n not in f["people"]:
                f["people"].append(n)
    known_places = {(r["place"] or {}).get("value", "") for r in rows}
    known_places |= {(r["event"] or {}).get("value", "") for r in rows}
    for pl in known_places:
        for tok in re.split(r"[,/]", pl):
            tok = tok.strip().lower()
            if len(tok) > 3 and tok in ql:
                f["places"].append(tok)
    f["years"], f["months"], f["places"] = list(set(f["years"])), list(set(f["months"])), list(set(f["places"]))
    return f


def _passes(r, f):
    if f["type"] and r["type"] != f["type"]:
        return False
    if f["years"] or f["months"]:
        if not r["taken_at"]:
            return False
        dt = datetime.fromisoformat(r["taken_at"])
        if f["years"] and dt.year not in f["years"]:
            return False
        if f["months"] and dt.month not in f["months"]:
            return False
    if f["people"]:
        names = {p["name"].lower() for p in (r["people"] or [])}
        if not all(p.lower() in names for p in f["people"]):
            return False
    if f["places"]:
        blob = " ".join([(r["place"] or {}).get("value", ""), (r["event"] or {}).get("value", ""),
                         " ".join(r["tags"] or [])]).lower()
        if not any(p in blob for p in f["places"]):
            return False
    return True


def _text(r):
    return " ".join([(r["caption"] or {}).get("value", ""), (r["event"] or {}).get("value", ""),
                     (r["place"] or {}).get("value", ""), " ".join(r["tags"] or []),
                     " ".join(p["name"] for p in (r["people"] or [])), r["ocr_text"] or ""]).lower()


def _minmax(a, floor=0.05):
    """Scale to 0..1, but never stretch a spread smaller than `floor` (tiny CLIP differences are mostly noise)."""
    a = np.asarray(a, dtype=np.float32)
    if len(a) == 0:
        return a
    return (a - a.min()) / max(float(a.max() - a.min()), floor)


def _remaining_words(q, f):
    """Words of the query NOT already used as filters (names, roles, places, years, months)."""
    used = set()
    for n in f["people"]:
        used |= set(re.findall(r"[a-z]+", n.lower()))
    for role in _roles():
        if re.search(rf"\b{re.escape(role)}\b", q.lower()):
            used.add(role)
    for pl in f["places"]:
        used |= set(re.findall(r"[a-z]+", pl))
    used |= {str(y) for y in f["years"]}
    return [w for w in re.findall(r"[a-z0-9]+", q.lower()) if w not in used and w not in MONTHS]


def reason(r, note=""):
    parts, src = [], []
    if r["people"]:
        parts.append(", ".join(p["name"] for p in r["people"]))
        src.append(SOURCE_LABEL.get(r["people"][0]["source"], r["people"][0]["source"]))
    if r["place"]:
        parts.append(r["place"]["value"]); src.append(SOURCE_LABEL.get(r["place"]["source"], r["place"]["source"]))
    elif r["event"]:
        parts.append(r["event"]["value"]); src.append(SOURCE_LABEL.get(r["event"]["source"], r["event"]["source"]))
    if r["taken_at"] and (r["taken_at_source"] or "").startswith("filesystem"):
        parts.append("date unknown (only file time available)")  # never present a download date as a fact
    elif r["taken_at"]:
        parts.append(datetime.fromisoformat(r["taken_at"]).strftime("%d %b %Y"))
        src.append(SOURCE_LABEL.get(r["taken_at_source"], r["taken_at_source"]))
    out = " · ".join(parts) + (f"  — from {' + '.join(dict.fromkeys(src))}" if src else "")
    if r["metadata_status"] == "stripped":
        out += "  (metadata stripped)"
    if (r["event"] or {}).get("source") == "vision_model_guess":
        out += "  (event is a vision guess)"
    return (out + (" " + note if note else "")).strip()


def search(q: str, limit: int = 12):
    conn = db.connect()
    C = _load(conn)
    rows = C["rows"]
    if not rows:
        return {"query": q, "filters": {}, "results": [], "note": "index is empty - run `index` first"}
    f = parse_query(q, rows)
    cands = [r for r in rows if _passes(r, f)]
    note = ""
    if not cands:  # soft fallback: semantic only
        cands, note = rows, "no exact metadata match - showing closest semantic matches"
    kept = _remaining_words(q, f)  # e.g. "Sanjay Agarwal in brown suit" -> ["in", "brown", "suit"]
    qtokens = [t for t in kept if t not in STOP and len(t) > 1]

    kw = np.array([sum(any(x in _text(r) for x in SYN.get(t, [t])) for t in qtokens) / max(len(qtokens), 1) for r in cands], dtype=np.float32)
    sim_img = np.zeros(len(cands), dtype=np.float32)
    sim_txt = np.zeros(len(cands), dtype=np.float32)
    semantic = False
    qv = embed.embed_text("a photo of a person, " + " ".join(kept)) if (qtokens and embed.available()) else None
    if qv is not None:
        semantic = True
        for i, r in enumerate(cands):
            iv, tv = C["vecs"][r["file_id"]]
            sim_img[i] = float(iv @ qv) if iv is not None else 0.0
            sim_txt[i] = float(tv @ qv) if tv is not None else 0.0
    if semantic:
        score = 0.5 * _minmax(sim_img) + 0.2 * _minmax(sim_txt) + 0.3 * kw
    elif qtokens:
        score = kw
    else:  # query was only filters (e.g. just a name): everything that passed matches equally well
        score = np.ones(len(cands), dtype=np.float32)
    order = np.argsort(-score)[:limit]
    results = []
    for i in order:
        r = cands[i]
        results.append({
            "file_id": r["file_id"], "filename": Path(r["source_path"]).name, "type": r["type"],
            "score": round(float(score[i]), 3), "caption": (r["caption"] or {}).get("value"),
            "taken_at": r["taken_at"], "place": (r["place"] or {}).get("value"),
            "people": [p["name"] for p in (r["people"] or [])], "event": (r["event"] or {}).get("value"),
            "tags": (r["tags"] or [])[:10], "metadata_status": r["metadata_status"],
            "flags": r["flags"], "reason": reason(r), "url": f"/api/file/{r['file_id']}",
        })
    return {"query": q, "filters": {k: v for k, v in f.items() if v}, "semantic": embed.available(),
            "note": note, "results": results}
