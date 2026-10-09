"""Metadata extraction + reconciliation (spec sections 3.1 - 3.4).

Rule: metadata FIRST, vision later. Every field carries a `source` and `confidence`.
"""
import hashlib
import json
import mimetypes
import os
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import config

IST = timezone(timedelta(hours=5, minutes=30))
GENERIC_SEGMENTS = {
    "photos", "images", "pictures", "dcim", "camera", "downloads", "desktop", "documents",
    "data", "users", "home", "mnt", "media", "nas", "share", "events", "whatsapp images",
    "whatsapp", "sample_photos", "people", "persons", "person", "organised", "misc", "new folder",
}


# ----------------------------------------------------------------- helpers
def shutil_which(cmd):
    import shutil
    return shutil.which(cmd)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in config.IMAGE_EXT:
        return "image"
    if ext in config.DOC_EXT:
        return "document"
    return "other"


def run_exiftool(path: Path):
    """Full dump: all groups, all tags (incl. unknown), numeric values. Falls back to Pillow."""
    from . import fallback_meta
    if config.FORCE_FALLBACK:
        return fallback_meta.read(path), "pure_python"
    try:
        out = subprocess.run(
            [config.EXIFTOOL_PATH, "-j", "-G", "-a", "-u", "-n", "-api", "largefilesupport=1", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        data = json.loads(out.stdout)[0]
        return data, "exiftool"
    except FileNotFoundError:
        if not getattr(run_exiftool, "_warned", False):
            import sys
            print("[exiftool not found -> using built-in pure-Python reader (no MakerNotes/video/HEIC)]", file=sys.stderr)
            run_exiftool._warned = True
        return fallback_meta.read(path), "pure_python"
    except Exception as e:  # corrupt / encrypted file etc.
        return {"Error": str(e)}, "error"


def _pillow_fallback(path: Path) -> dict:
    """Only used if exiftool is missing. MISSES IPTC/XMP/region tags -> install exiftool!"""
    meta = {}
    try:
        from PIL import Image
        ex = Image.open(path).getexif()
        names = {0x010F: "Make", 0x0110: "Model", 0x9003: "DateTimeOriginal", 0x0132: "ModifyDate"}
        sub = ex.get_ifd(0x8769)
        for tag, name in names.items():
            v = ex.get(tag) or sub.get(tag)
            if v:
                meta[f"EXIF:{name}"] = v
    except Exception:
        pass
    return meta


def _norm(k: str) -> str:
    """exiftool -G prints family-0 groups ("XMP:Subject"); sub-groups like XMP-dc/XMP-lr collapse to XMP."""
    return re.sub(r"^XMP-[A-Za-z\-]+:", "XMP:", k)


def first(meta: dict, *keys):
    for k in keys:
        v = meta.get(_norm(k))
        if v not in (None, "", [], "0000:00:00 00:00:00"):
            return v
    return None


def by_suffix(meta: dict, suffix: str):
    for k, v in meta.items():
        if k.endswith(suffix) and v not in (None, "", []):
            return v
    return None


def as_list(v):
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


_DT = re.compile(r"(\d{4})[:\-](\d{2})[:\-](\d{2})[ T](\d{2}):(\d{2}):(\d{2})(?:\.\d+)?\s*(Z|[+-]\d{2}:?\d{2})?")
_FN = re.compile(r"(?<!\d)((?:19|20)\d{2})[-_]?(0[1-9]|1[0-2])[-_]?(0[1-9]|[12]\d|3[01])"
                 r"(?:[ _\-T]?(?:at )?([01]\d|2[0-3])[.:_]?([0-5]\d)[.:_]?([0-5]\d))?")


def _tz(s):
    if s is None:
        return None
    if s == "Z":
        return timezone.utc
    m = re.match(r"([+-])(\d{2}):?(\d{2})", s)
    if not m:
        return None
    delta = timedelta(hours=int(m[2]), minutes=int(m[3]))
    return timezone(delta if m[1] == "+" else -delta)


def parse_dt(s, offset=None):
    """-> (datetime|None, tz_assumed:bool)"""
    if not isinstance(s, str):
        return None, False
    m = _DT.search(s)
    if not m or m[1] == "0000":
        return None, False
    tz = _tz(m[7] or (offset if isinstance(offset, str) else None))
    assumed = tz is None
    try:
        dt = datetime(*(int(x) for x in m.groups()[:6]), tzinfo=tz or IST)
    except ValueError:
        return None, False
    return dt, assumed


def date_from_filename(name: str):
    m = _FN.search(name)
    if not m:
        return None
    y, mo, d, h, mi, s = m.groups()
    try:
        return datetime(int(y), int(mo), int(d), int(h or 0), int(mi or 0), int(s or 0), tzinfo=IST)
    except ValueError:
        return None


def read_takeout(path: Path):
    for cand in (Path(str(path) + ".json"), Path(str(path) + ".supplemental-metadata.json")):
        if cand.exists():
            try:
                return json.loads(cand.read_text(encoding="utf-8"))
            except Exception:
                pass
    return None


# ------------------------------------------------------------ reconcilers
def reconcile_date(meta, path: Path, takeout):
    """Trust order (3.4): EXIF DTO(+offset) > GPS ts > Takeout > QuickTime > XMP > filename > PDF/Office > fs."""
    flags = []
    cands = []  # (datetime, source, confidence, tz_source)

    dt, assumed = parse_dt(first(meta, "EXIF:DateTimeOriginal"), first(meta, "EXIF:OffsetTimeOriginal", "EXIF:OffsetTime"))
    if dt:
        cands.append((dt, "exif_datetimeoriginal", 0.95 if assumed else 0.99,
                      "assumed_asia_kolkata" if assumed else "exif_offsettime"))
    dt, _ = parse_dt(first(meta, "Composite:GPSDateTime"))
    if dt:
        cands.append((dt.astimezone(IST), "gps_timestamp", 0.90, "utc_converted"))
    if takeout and takeout.get("photoTakenTime", {}).get("timestamp"):
        try:
            dt = datetime.fromtimestamp(int(takeout["photoTakenTime"]["timestamp"]), tz=timezone.utc).astimezone(IST)
            cands.append((dt, "takeout_json_phototakentime", 0.90, "utc_converted"))
        except Exception:
            pass
    dt, _ = parse_dt(first(meta, "QuickTime:CreateDate", "QuickTime:MediaCreateDate"))
    if dt:
        cands.append((dt.astimezone(IST), "quicktime_createdate", 0.80, "assumed_utc"))
    dt, assumed = parse_dt(first(meta, "XMP:CreateDate", "XMP-xmp:CreateDate", "EXIF:CreateDate"))
    if dt:
        cands.append((dt, "xmp_createdate", 0.80, "assumed_asia_kolkata" if assumed else "embedded"))
    dt = date_from_filename(path.name)
    if dt:
        cands.append((dt, "filename_embedded_date", 0.70, "assumed_asia_kolkata"))
    dt, assumed = parse_dt(first(meta, "PDF:CreateDate", "XML:CreateDate", "XMP-dcterms:Created"))
    if dt:
        cands.append((dt, "doc_creation_date", 0.75, "assumed_asia_kolkata" if assumed else "embedded"))

    st = path.stat()
    birth = getattr(st, "st_birthtime", None)
    if birth:
        cands.append((datetime.fromtimestamp(birth, tz=IST), "filesystem_birthtime", 0.30, "assumed_asia_kolkata"))
    cands.append((datetime.fromtimestamp(st.st_mtime, tz=IST), "filesystem_mtime", 0.15, "assumed_asia_kolkata"))

    best = cands[0]  # list is already in trust order; fs entries are last
    if best[1].startswith("filesystem"):
        flags.append("date_unreliable_fs_only")
    real = [c for c in cands if not c[1].startswith("filesystem")]  # fs dates are never a fair comparison
    if len(real) > 1 and abs((real[0][0] - real[1][0]).total_seconds()) > 86400:
        flags.append("date_sources_disagree")

    # Sanity checks
    now = datetime.now(IST)
    if best[0].year < 1990 or best[0] > now + timedelta(days=1):
        flags.append("suspect_date")
    mod, _ = parse_dt(first(meta, "EXIF:ModifyDate"))
    cre, _ = parse_dt(first(meta, "EXIF:CreateDate"))
    if mod and cre and mod < cre:
        flags.append("file_touched")
    return best, flags


def reverse_geo(lat, lon):
    try:
        import reverse_geocoder as rg
        r = rg.search((lat, lon), mode=1)[0]
        return ", ".join(x for x in (r.get("name"), r.get("admin1"), r.get("cc")) if x)
    except Exception:
        return None


def reconcile_place(meta, takeout):
    lat = first(meta, "Composite:GPSLatitude", "EXIF:GPSLatitude")
    lon = first(meta, "Composite:GPSLongitude", "EXIF:GPSLongitude")
    err = first(meta, "EXIF:GPSHPositioningError")
    src = "exif_gps"
    if lat is None and takeout:
        g = takeout.get("geoData") or {}
        lat, lon, src = g.get("latitude"), g.get("longitude"), "takeout_json_geodata"
    flags = []
    if lat is not None and lon is not None:
        try:
            lat, lon = float(lat), float(lon)
        except (TypeError, ValueError):
            lat = lon = None
        if lat == 0 and lon == 0:  # sanity: GPS at exactly 0,0 -> null it
            lat = lon = None
            flags.append("gps_zero_nulled")
    # Human-written place wins on specificity (IPTC sub-location), GPS gives coordinates
    iptc_parts = [first(meta, k) for k in ("IPTC:Sub-location", "IPTC:City", "XMP-photoshop:City",
                                           "IPTC:Province-State", "XMP-photoshop:State",
                                           "IPTC:Country-PrimaryLocationName", "XMP-photoshop:Country")]
    iptc_parts = [str(p) for p in iptc_parts if p]
    if lat is not None and lon is not None:
        geo = reverse_geo(lat, lon)
        value = ", ".join(iptc_parts[:2] + ([geo] if geo and not iptc_parts else [])) or geo or f"{lat:.4f},{lon:.4f}"
        return {"value": value, "source": src + ("+iptc" if iptc_parts else "+reverse_geo"),
                "confidence": 0.9, "lat": lat, "lon": lon, "error_m": err}, flags
    if iptc_parts:
        return {"value": ", ".join(iptc_parts), "source": "iptc_xmp_location", "confidence": 0.85}, flags
    return None, flags


def reconcile_people(meta, takeout):
    names = []
    for key_suffix in (":RegionName", ":RegionPersonDisplayName", ":PersonInImage"):
        for n in as_list(by_suffix(meta, key_suffix)):
            names.append((str(n), "metadata_region", 0.99))
    if takeout:
        for p in takeout.get("people", []):
            if p.get("name"):
                names.append((p["name"], "takeout_json_people", 0.90))
    seen, out = set(), []
    for n, s, c in names:
        if n.lower() not in seen:
            seen.add(n.lower())
            out.append({"name": n, "source": s, "confidence": c})
    return out


def keywords(meta):
    kws = []
    for k in ("IPTC:Keywords", "XMP-dc:Subject", "XMP:Subject", "XMP-lr:HierarchicalSubject",
              "XMP-MicrosoftPhoto:LastKeywordXMP", "EXIF:XPKeywords"):
        kws += [str(x) for x in as_list(meta.get(_norm(k)))]
    flat = []
    for k in kws:
        flat += [p.strip() for p in re.split(r"[|;]", k) if p.strip()]
    return list(dict.fromkeys(flat))


FN_STOP = {"sir", "madam", "folder", "images", "photos", "pics", "pictures", "files", "new", "img", "image", "photo", "jpg", "jpeg", "png", "whatsapp", "screenshot", "dsc", "pxl", "copy", "final"}


def filename_tokens(path: Path):
    return [t.lower() for t in re.findall(r"[A-Za-z]{3,}", path.stem) if t.lower() not in FN_STOP]


def name_hints(path: Path, rel: Path):
    """Names via people.json aliases from filename (0.6) and folder names like 'Sanjay Sir' (0.4).
    Low trust: a filename/folder is a hint, not proof. Real region tags always win."""
    try:
        lib = json.loads(config.PEOPLE_FILE.read_text())
    except Exception:
        return []
    out, seen = [], set()
    folder_toks = {t.lower() for part in rel.parent.parts for t in re.findall(r"[A-Za-z]{3,}", part)}
    for source, conf, toks in (("filename_hint", 0.6, set(filename_tokens(path))), ("folder_hint", 0.4, folder_toks)):
        for n, aliases in lib.items():
            if n not in seen and toks & {a.lower() for a in aliases}:
                seen.add(n)
                out.append({"name": n, "source": source, "confidence": conf})
    return out


def _person_alias_tokens():
    try:
        return {a.lower() for al in json.loads(config.PEOPLE_FILE.read_text()).values() for a in al}
    except Exception:
        return set()


HONORIFICS = {"sir", "madam", "mam", "maam", "ji", "shri", "sri", "mr", "mrs", "ms", "dr", "smt", "photos", "images", "pics"}


def owner_folder(rel_in_root: Path, root: Path):
    """Which folder name stands for 'whose folder is this' (None unless PERSON_FOLDERS is on)."""
    if config.PERSON_FOLDERS == "child" and len(rel_in_root.parts) > 1:
        return rel_in_root.parts[0]
    if config.PERSON_FOLDERS == "root":
        return root.name
    return None


def clean_person_name(folder: str):
    toks = [t for t in re.findall(r"[A-Za-z]+", folder) if t.lower() not in HONORIFICS]
    return " ".join(t.capitalize() for t in toks) or None


def reconcile_event(meta, rel_path: Path, skip_segments=()):
    hier = as_list(first(meta, "XMP-lr:HierarchicalSubject"))
    for h in hier:
        parts = [p.strip() for p in str(h).split("|")]
        if len(parts) >= 2 and parts[0].lower() in ("events", "event"):
            return {"value": " / ".join(parts[1:]), "source": "hierarchical_keyword", "confidence": 0.95}
    people_tokens = _person_alias_tokens()
    for seg in reversed(rel_path.parent.parts):  # folder path is high-signal
        seg_tokens = {t.lower() for t in re.findall(r"[A-Za-z]{3,}", seg)}
        if seg_tokens & people_tokens or seg in skip_segments:  # "Sanjay Sir" is a person's folder, not an event
            continue
        if seg.lower() not in GENERIC_SEGMENTS and not re.fullmatch(r"[\d\-_. ]+", seg):
            return {"value": seg, "source": "source_path", "confidence": 0.70}
    return None


def reconcile_caption(meta, takeout):
    v = first(meta, "IPTC:Caption-Abstract", "XMP-dc:Description", "EXIF:ImageDescription",
              "EXIF:UserComment", "XMP-dc:Title", "IPTC:Headline", "XMP-photoshop:Headline")
    if v:
        return {"value": str(v), "source": "embedded_caption", "confidence": 0.95}
    if takeout and takeout.get("description"):
        return {"value": takeout["description"], "source": "takeout_json_description", "confidence": 0.9}
    return None


_REAL_GROUPS = ("EXIF", "IPTC", "XMP", "MakerNotes", "GPS", "QuickTime", "PDF", "XML", "ICC")


def metadata_status(meta):
    real = [k for k in meta if k.split(":")[0].startswith(_REAL_GROUPS)
            and not k.startswith(("ICC_Profile", "ExifTool"))]
    has_core = bool(first(meta, "EXIF:DateTimeOriginal", "PDF:CreateDate", "XML:CreateDate")) and bool(
        first(meta, "EXIF:Make", "Composite:GPSLatitude", "PDF:Producer", "XML:Creator", "PDF:Author"))
    if not real:
        return "stripped"
    return "complete" if has_core else "partial"


def pdf_text(path: Path, max_pages=3) -> str:
    try:
        from pypdf import PdfReader
        r = PdfReader(str(path))
        if r.is_encrypted:
            return ""
        return "\n".join((p.extract_text() or "") for p in r.pages[:max_pages]).strip()
    except Exception:
        return ""


# ------------------------------------------------------------------ main
def extract(path: Path, root: Path) -> dict:
    path = Path(path)
    rel = path.relative_to(root.parent) if path.is_relative_to(root) else Path(path.name)  # includes root's own name
    rel_in_root = path.relative_to(root) if path.is_relative_to(root) else Path(path.name)  # for event: root name is just a container
    meta, tool = run_exiftool(path)
    meta = {_norm(k): v for k, v in meta.items()}
    takeout = read_takeout(path)
    ftype = file_type(path)

    (dt, d_src, d_conf, tz_src), flags = reconcile_date(meta, path, takeout)
    place, pflags = reconcile_place(meta, takeout)
    flags += pflags
    if tool != "exiftool":  # e.g. pure_python
        flags.append(f"extraction_{tool}")
    if first(meta, "PDF:Encryption", "PDF:Encrypt") or meta.get("Error"):
        flags.append("encrypted_or_corrupt")

    kws = keywords(meta)
    path_tags = [s for s in rel.parent.parts if s]  # every path segment is a candidate tag
    tags = list(dict.fromkeys(kws + path_tags + filename_tokens(path)))
    people = reconcile_people(meta, takeout)
    known = {p["name"].lower() for p in people}
    people += [h for h in name_hints(path, rel) if h["name"].lower() not in known]  # never override real tags
    owner = owner_folder(rel_in_root, root)
    if owner and not any(h["source"] == "folder_hint" for h in people):  # auto person from folder name
        nm = clean_person_name(owner)
        if nm and nm.lower() not in {p["name"].lower() for p in people}:
            people.append({"name": nm, "source": "folder_owner_hint", "confidence": 0.4})

    return {
        "type": ftype,
        "mime": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        "taken_at": dt.isoformat(),
        "taken_at_source": d_src,
        "taken_at_confidence": d_conf,
        "timezone_source": tz_src,
        "place": place,
        "people": people,
        "event": reconcile_event(meta, rel_in_root, {owner} if owner else ()),
        "caption": reconcile_caption(meta, takeout),
        "tags": tags,
        "ocr_text": pdf_text(path) if ftype == "document" else None,
        "metadata_status": metadata_status(meta),
        "flags": flags,
        "rating": first(meta, "XMP:Rating", "XMP-xmp:Rating", "EXIF:Rating"),
        "camera": {
            "make": first(meta, "EXIF:Make"), "model": first(meta, "EXIF:Model"),
            "serial": first(meta, "EXIF:SerialNumber", "MakerNotes:SerialNumber", "EXIF:BodySerialNumber"),
            "lens": first(meta, "EXIF:LensModel", "Composite:LensID"),
        },
        "raw_metadata": meta,  # full dump goes into the index (3: "extract everything")
    }
