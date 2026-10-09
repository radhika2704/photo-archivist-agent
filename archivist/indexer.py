"""Scan source -> metadata -> vision -> embeddings -> index.db. Source tree is READ-ONLY."""
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

from . import colour, config, db, embed, metadata, vision


def iter_files(root: Path):
    excl = [Path(e).resolve() for e in config.EXCLUDE] + [config.DATA_DIR]
    for dp, dns, fns in os.walk(root):
        dp = Path(dp).resolve()
        dns[:] = [d for d in dns if (dp / d).resolve() not in excl and not d.startswith(".")]
        for fn in sorted(fns):
            p = dp / fn
            if p.suffix.lower() in config.SUPPORTED:
                yield p


def dry_run(root: Path, sample: int = 300):
    """Spec section 9: writes NOTHING. Prints metadata coverage statistics."""
    files = list(iter_files(root))[:sample]
    c = Counter()
    seen, dups = {}, []
    print(f"DRY RUN over {len(files)} files from {root}\n")
    for p in files:
        rec = metadata.extract(p, root)
        m = rec["raw_metadata"]
        c["files"] += 1
        c[f"type:{rec['type']}"] += 1
        c["has_exif"] += any(k.startswith("EXIF:") for k in m)
        c["has_gps"] += rec["place"] is not None and rec["place"].get("lat") is not None
        c["has_people_region_tags"] += bool(rec["people"])
        c["has_keywords"] += bool([k for k in m if "Keywords" in k or "Subject" in k])
        c["has_text_layer"] += bool(rec["ocr_text"])
        c["stripped"] += rec["metadata_status"] == "stripped"
        c[f"date_source:{rec['taken_at_source']}"] += 1
        for f in rec["flags"]:
            c[f"flag:{f}"] += 1
        h = metadata.sha256_file(p)
        if h in seen:
            dups.append((str(p), seen[h]))
        seen[h] = str(p)
        tags = rec["tags"][:6]
        print(f"- {p.name}: date={rec['taken_at'][:10]} ({rec['taken_at_source']}) "
              f"place={(rec['place'] or {}).get('value')} people={[x['name'] for x in rec['people']]} "
              f"status={rec['metadata_status']} tags={tags}")
    print("\n=== COVERAGE ===")
    for k, v in sorted(c.items()):
        print(f"{k:40s} {v}")
    print(f"\nduplicates found: {len(dups)}")
    for d in dups:
        print("  ", d[0], "==", d[1])
    print("\nNothing was written. Run `python -m archivist.cli index` after approval.")


def index_all(root: Path, use_vision=True, use_embed=True, limit=None):
    conn = db.connect()
    files = list(iter_files(root))[:limit]
    print(f"Indexing {len(files)} files (vision={'on' if use_vision and config.USE_VISION else 'off'}, "
          f"embeddings={'on' if use_embed else 'off'})")
    new = dup = 0
    for i, p in enumerate(files, 1):
        sha = metadata.sha256_file(p)
        fid = sha[:16]
        if conn.execute("SELECT 1 FROM files WHERE file_id=?", (fid,)).fetchone():
            if conn.execute("SELECT source_path FROM files WHERE file_id=?", (fid,)).fetchone()[0] != str(p):
                conn.execute("INSERT OR IGNORE INTO duplicates VALUES (?,?)", (fid, str(p)))  # report, never remove
                dup += 1
            continue
        rec = metadata.extract(p, root)
        va = vision.analyse(p) if (use_vision and rec["type"] == "image") else None
        if va:
            if not rec["caption"] and va.get("caption"):
                rec["caption"] = {"value": va["caption"], "source": "vision_model", "confidence": 0.6}
            rec["tags"] = list(dict.fromkeys(rec["tags"] + [t.lower() for t in va.get("tags", []) + va.get("attributes", [])]))
            rec["ocr_text"] = va.get("ocr_text") or rec["ocr_text"]
            if not rec["event"] and va.get("event_guess"):
                rec["event"] = {"value": va["event_guess"], "source": "vision_model_guess", "confidence": 0.5}
            if va.get("pii_visible"):
                rec["flags"].append("pii_visible")
        if rec["metadata_status"] == "stripped":
            rec["flags"].append("metadata_stripped_leaning_on_vision_and_path")
        if rec["type"] == "image":  # offline colour hint (tagged as an estimate)
            rec["tags"] = list(dict.fromkeys(rec["tags"] + [f"est. {c} clothing" for c in colour.clothing_colours(p)]))

        img_vec = txt_vec = None
        if use_embed and embed.available():
            if rec["type"] == "image":
                img_vec = embed.embed_image(p)
            txt_vec = embed.embed_text(_text_blob(rec))

        conn.execute(
            """INSERT INTO files(file_id,sha256,source_path,type,mime,taken_at,taken_at_source,taken_at_confidence,
               timezone_source,place,people,event,caption,tags,ocr_text,metadata_status,flags,rating,camera,
               raw_metadata,img_vec,txt_vec,indexed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (fid, sha, str(p), rec["type"], rec["mime"], rec["taken_at"], rec["taken_at_source"],
             rec["taken_at_confidence"], rec["timezone_source"], json.dumps(rec["place"]),
             json.dumps(rec["people"]), json.dumps(rec["event"]), json.dumps(rec["caption"]),
             json.dumps(rec["tags"]), rec["ocr_text"], rec["metadata_status"], json.dumps(rec["flags"]),
             int(rec["rating"]) if str(rec["rating"] or "").lstrip("-").isdigit() else None,
             json.dumps(rec["camera"]), json.dumps(rec["raw_metadata"], default=str),
             embed.to_blob(img_vec), embed.to_blob(txt_vec), datetime.now().isoformat()),
        )
        conn.commit()
        new += 1
        print(f"[{i}/{len(files)}] {p.name}  date={rec['taken_at'][:10]}({rec['taken_at_source']}) "
              f"status={rec['metadata_status']}")
    conn.commit()  # duplicates are inserted without a commit above - flush them
    print(f"\nDone. new={new} duplicates_reported={dup}")


def _text_blob(rec: dict) -> str:
    parts = [(rec["caption"] or {}).get("value"), (rec["event"] or {}).get("value"),
             (rec["place"] or {}).get("value"), " ".join(rec["tags"]),
             " ".join(p["name"] for p in rec["people"]), (rec["ocr_text"] or "")[:300]]
    return ". ".join(p for p in parts if p)
