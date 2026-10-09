import argparse
from pathlib import Path

from . import config, db, indexer, organiser


def main():
    ap = argparse.ArgumentParser(prog="archivist")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("dry-run", help="coverage report, writes nothing"); p.add_argument("--sample", type=int, default=300)
    p = sub.add_parser("index", help="build index.db")
    p.add_argument("--no-vision", action="store_true"); p.add_argument("--no-embed", action="store_true")
    p.add_argument("--limit", type=int)
    p = sub.add_parser("organise", help="plan folder decisions (add --apply to copy)"); p.add_argument("--apply", action="store_true")
    sub.add_parser("undo", help="reverse last organise batch")
    p = sub.add_parser("serve"); p.add_argument("--port", type=int, default=8000)
    p = sub.add_parser("search"); p.add_argument("q")
    p = sub.add_parser("inspect", help="show caption/tags/OCR/embeddings stored for files")
    p.add_argument("name", nargs="?", default="", help="part of filename (empty = all)")
    p.add_argument("--vectors", action="store_true", help="print the full vectors")
    a = ap.parse_args()

    if a.cmd == "dry-run":
        for root in config.SOURCE_DIRS:
            indexer.dry_run(root, a.sample)
    elif a.cmd == "index":
        for root in config.SOURCE_DIRS:
            print(f"\n##### root: {root}")
            indexer.index_all(root, use_vision=not a.no_vision, use_embed=not a.no_embed, limit=a.limit)
    elif a.cmd == "organise":
        conn = db.connect()
        acts = organiser.plan(conn)
        for x in acts:
            print(f"{x['action']:7s} -> {x['folder']:40s} {Path(x['src']).name:28s} {x['why']}")
        if a.apply:
            b, n = organiser.apply(conn, acts)
            print(f"\napplied {n} copies as {b}. Undo with: python -m archivist.cli undo")
        else:
            print(f"\nDRY RUN: {len(acts)} planned actions, nothing written. Re-run with --apply after approval.")
    elif a.cmd == "undo":
        organiser.undo_last(db.connect())
    elif a.cmd == "serve":
        import uvicorn
        uvicorn.run("archivist.app:app", host="0.0.0.0", port=a.port)
    elif a.cmd == "inspect":
        import json
        from . import embed
        rows = db.connect().execute("SELECT * FROM files WHERE source_path LIKE ?", (f"%{a.name}%",)).fetchall()
        if not rows:
            print("no matching files in index (run `index` first)")
        for r in rows:
            d = db.row_to_dict(r)
            cap = d["caption"] or {}
            print("=" * 78)
            print("file      :", Path(d["source_path"]).name, f"  [{d['type']}, metadata={d['metadata_status']}]")
            print("caption   :", cap.get("value") or "(none - vision is off or failed)", f"  <- source: {cap.get('source')}")
            print("tags      :", ", ".join(d["tags"] or []))
            print("ocr_text  :", (d["ocr_text"] or "")[:200] or "(none)")
            print("event     :", (d["event"] or {}).get("value"), "|", (d["event"] or {}).get("source"))
            print("people    :", [(p["name"], p["source"], p["confidence"]) for p in d["people"] or []])
            print("date      :", d["taken_at"], "<-", d["taken_at_source"])
            for col in ("img_vec", "txt_vec"):
                v = embed.from_blob(r[col])
                if v is None:
                    print(f"{col:10s}: (none - indexed with --no-embed or no image)")
                else:
                    print(f"{col:10s}: {len(v)} dims, norm={float((v**2).sum()**.5):.3f}, first 8 = {[round(float(x), 4) for x in v[:8]]}")
                    if a.vectors:
                        print(json.dumps([round(float(x), 5) for x in v]))
    elif a.cmd == "search":
        from . import search
        import json
        print(json.dumps(search.search(a.q), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
