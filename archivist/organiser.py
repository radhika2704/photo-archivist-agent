"""Organise a COPY of the library (spec section 5). Dry-run by default, every write is logged and undoable."""
import json
import re
import shutil
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from . import config, db, embed

REVIEW = "_Review"


def _profile_add(prof, row, vec):
    prof["vecs"].append(vec) if vec is not None else None
    prof["tags"].update(t.lower() for t in (row["tags"] or []))
    prof["people"].update(p["name"] for p in (row["people"] or []))
    prof["events"].add(((row["event"] or {}).get("value") or "").lower())
    if row["taken_at"]:
        d = datetime.fromisoformat(row["taken_at"])
        prof["min"], prof["max"] = min(prof["min"] or d, d), max(prof["max"] or d, d)


def _new_profile():
    return {"vecs": [], "tags": set(), "people": set(), "events": set(), "min": None, "max": None}


def build_profiles(conn):
    profs = defaultdict(_new_profile)
    for r in conn.execute("SELECT * FROM files WHERE organised_path IS NOT NULL"):
        d = db.row_to_dict(r)
        folder = Path(d["organised_path"]).parent.name
        if folder == REVIEW:
            continue
        _profile_add(profs[folder], d, embed.from_blob(r["img_vec"]))
    return profs


def score_against(row, vec, prof):
    log = []
    centroid = np.mean(prof["vecs"], axis=0) if prof["vecs"] else None
    has_vec = centroid is not None and vec is not None
    if has_vec:
        c = centroid / (np.linalg.norm(centroid) + 1e-9)
        vs = max(0.0, float(vec @ c))
    else:
        vs = 0.0
    tags = {t.lower() for t in (row["tags"] or [])}
    tag_ov = len(tags & prof["tags"]) / len(tags) if tags else 0.0
    people = {p["name"] for p in (row["people"] or [])}
    if not people and not prof["people"]:
        people_ov = 1.0  # neither side has people -> neutral, don't punish
    else:
        people_ov = len(people & prof["people"]) / len(people | prof["people"]) if (people | prof["people"]) else 0.0
    score = 0.50 * vs + 0.30 * tag_ov + 0.20 * people_ov
    if not has_vec:  # embeddings off -> renormalise tag+people weights so thresholds stay meaningful
        score = (0.30 * tag_ov + 0.20 * people_ov) / 0.50
        log.append("no vectors: tag+people only")
    # metadata boost: same event keyword + date inside folder span (hard metadata match)
    ev = ((row["event"] or {}).get("value") or "").lower()
    if ev and ev in prof["events"] and row["taken_at"] and prof["min"]:
        d = datetime.fromisoformat(row["taken_at"])
        if prof["min"] - timedelta(days=2) <= d <= prof["max"] + timedelta(days=2):
            score += 0.15
            log.append("boost: same event keyword + date within folder span")
    return score, log


def new_folder_name(row):
    d = datetime.fromisoformat(row["taken_at"])
    label = (row["event"] or {}).get("value") or ((row["place"] or {}).get("value") or "").split(",")[0] or "Unsorted"
    label = re.sub(r'[\\/:*?"<>|]', "-", label).strip()[:60]
    return f"{d:%Y-%m} {label}"  # pattern: "YYYY-MM Event"; change here to match your drive's naming


def plan(conn):
    rows = [r for r in conn.execute("SELECT * FROM files WHERE organised_path IS NULL ORDER BY taken_at")]
    profs = build_profiles(conn)
    actions = []
    for r in rows:
        d = db.row_to_dict(r)
        vec = embed.from_blob(r["img_vec"])
        flags = d["flags"] or []
        unreliable = "suspect_date" in flags or "date_unreliable_fs_only" in flags
        if d["type"] == "document":
            year = d["taken_at"][:4] if not unreliable else "Undated"
            actions.append({"file_id": d["file_id"], "src": d["source_path"], "action": "copy",
                            "folder": f"Documents/{year}", "score": None, "why": "documents go by year"})
            continue
        cands = sorted(((score_against(d, vec, p), name) for name, p in profs.items()),
                       key=lambda x: -x[0][0])
        top = [(round(s[0], 2), n, s[1]) for s, n in cands[:2]]
        if top and top[0][0] >= config.THRESH_HIGH:
            act, folder, why = "copy", top[0][1], f"score {top[0][0]:.2f} >= {config.THRESH_HIGH}" + (
                f" ({'; '.join(top[0][2])})" if top[0][2] else "")
        elif top and top[0][0] >= config.THRESH_LOW:
            act, folder, why = "review", REVIEW, f"score {top[0][0]:.2f} borderline -> ask user"
        elif unreliable:
            act, folder, why = "review", REVIEW, "would create a new dated folder but date is unreliable"
        else:
            act, folder = "new", new_folder_name(d)
            why = f"best score {top[0][0]:.2f} < {config.THRESH_LOW}" if top else "no folders yet"
            if folder in profs:  # same name already planned in this batch -> just join it
                act, why = "copy", f"joins '{folder}' (same event + date, name match)"
            _profile_add(profs[folder], d, vec)  # so later files in this batch can join it
        if act == "copy":
            _profile_add(profs[folder], d, vec)
        actions.append({"file_id": d["file_id"], "src": d["source_path"], "action": act, "folder": folder,
                        "score": top[0][0] if top else None, "candidates": [(n, s) for s, n, _ in top],
                        "why": why})
    return actions


def apply(conn, actions):
    batch = f"batch-{int(time.time())}"
    ops = []
    for a in actions:
        src = Path(a["src"])
        dst_dir = config.ORGANISED_DIR / a["folder"]
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        if dst.exists():
            dst = dst_dir / f"{src.stem}__{a['file_id'][:6]}{src.suffix}"
        shutil.copy2(src, dst)  # byte-for-byte copy: EXIF/XMP/IPTC preserved
        row = conn.execute("SELECT * FROM files WHERE file_id=?", (a["file_id"],)).fetchone()
        rec = db.row_to_dict(row, with_raw=True)
        rec["organised_path"] = str(dst)
        rec["folder_decision"] = {k: a[k] for k in ("action", "score", "why")}
        sidecar = Path(str(dst) + ".tags.json")
        sidecar.write_text(json.dumps(rec, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
        conn.execute("UPDATE files SET organised_path=? WHERE file_id=?", (str(dst), a["file_id"]))
        ops.append({"dst": str(dst), "sidecar": str(sidecar), "file_id": a["file_id"]})
    conn.commit()
    with open(config.UNDO_LOG, "a") as f:
        f.write(json.dumps({"batch": batch, "ops": ops}) + "\n")
    return batch, len(ops)


def undo_last(conn):
    if not config.UNDO_LOG.exists():
        print("nothing to undo"); return
    lines = config.UNDO_LOG.read_text().strip().splitlines()
    if not lines:
        print("nothing to undo"); return
    last = json.loads(lines[-1])
    for op in last["ops"]:
        for p in (op["dst"], op["sidecar"]):
            Path(p).unlink(missing_ok=True)  # only deletes OUR copies, never the source
        conn.execute("UPDATE files SET organised_path=NULL WHERE file_id=?", (op["file_id"],))
    conn.commit()
    config.UNDO_LOG.write_text("\n".join(lines[:-1]) + ("\n" if len(lines) > 1 else ""))
    print(f"undone {last['batch']} ({len(last['ops'])} files)")
