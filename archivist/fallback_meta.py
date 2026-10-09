"""exiftool-free metadata reader. Needs only Pillow + pypdf.

It returns the SAME key names exiftool -G would ("EXIF:DateTimeOriginal", "Composite:GPSLatitude",
"XMP-mwg-rs:RegionName", "IPTC:Keywords", "PDF:CreateDate" ...), so the rest of the pipeline
doesn't care which reader produced the dict.

Covers: EXIF (IFD0 + Exif + GPS), IPTC (JPEG/TIFF), XMP (people regions, keywords, hierarchical subjects,
description, city, rating), PDF info dict. Does NOT cover MakerNotes / video / HEIC (use exiftool for those).
"""
import re
from pathlib import Path

from PIL import ExifTags, Image, IptcImagePlugin

_TAGS = dict(ExifTags.TAGS)
_TAGS.update({0x9010: "OffsetTime", 0x9011: "OffsetTimeOriginal", 0x9012: "OffsetTimeDigitized",
              0xA434: "LensModel", 0xA431: "BodySerialNumber"})


def _clean(name, v):
    if isinstance(v, bytes):
        if name.startswith("XP"):
            return v.decode("utf-16-le", "ignore").rstrip("\x00")
        if name == "UserComment":
            v = v[8:] if v[:5] in (b"ASCII", b"UNICO", b"JIS\x00\x00") else v
        return v.decode("utf-8", "ignore").strip("\x00 ").strip()
    if hasattr(v, "numerator"):  # IFDRational
        try:
            return float(v)
        except Exception:
            return str(v)
    if isinstance(v, (tuple, list)):
        return [_clean(name, x) for x in v]
    return v


def _dms(vals, ref):
    d, m, s = (float(x) for x in vals)
    out = d + m / 60 + s / 3600
    return -out if ref in ("S", "W") else out


def _xmp_items(xml, tag):
    m = re.search(rf"<{tag}\b[^>]*>(.*?)</{tag}>", xml, re.S)
    return [re.sub(r"<[^>]+>", "", x).strip() for x in re.findall(r"<rdf:li[^>]*>(.*?)</rdf:li>", m.group(1), re.S)] if m else []


def _xmp_scalar(xml, tag):
    m = re.search(rf'{tag}\s*=\s*"([^"]*)"', xml) or re.search(rf"<{tag}\b[^>]*>([^<]+)</{tag}>", xml)
    return m.group(1).strip() if m else None


def _read_xmp(path: Path, meta: dict):
    try:
        with open(path, "rb") as f:
            raw = f.read(6 * 1024 * 1024)
        a = raw.find(b"<x:xmpmeta")
        if a < 0:
            return
        xml = raw[a:raw.find(b"</x:xmpmeta>", a) + 12].decode("utf-8", "ignore")
    except Exception:
        return
    names = re.findall(r'mwg-rs:Name\s*=\s*"([^"]+)"', xml) + re.findall(r"<mwg-rs:Name>([^<]+)</mwg-rs:Name>", xml)
    if names:
        meta["XMP-mwg-rs:RegionName"] = list(dict.fromkeys(names))
    for key, tag in (("XMP-iptcExt:PersonInImage", "Iptc4xmpExt:PersonInImage"), ("XMP-dc:Subject", "dc:subject"),
                     ("XMP-lr:HierarchicalSubject", "lr:hierarchicalSubject"), ("XMP-dc:Description", "dc:description"),
                     ("XMP-dc:Title", "dc:title")):
        items = _xmp_items(xml, tag)
        if items:
            meta[key] = items
    for key, tag in (("XMP-photoshop:City", "photoshop:City"), ("XMP-photoshop:State", "photoshop:State"),
                     ("XMP-photoshop:Country", "photoshop:Country"), ("XMP-xmp:CreateDate", "xmp:CreateDate"),
                     ("XMP-xmp:Rating", "xmp:Rating"), ("XMP-photoshop:Headline", "photoshop:Headline")):
        v = _xmp_scalar(xml, tag)
        if v:
            meta[key] = v


_IPTC = {25: "Keywords", 120: "Caption-Abstract", 90: "City", 92: "Sub-location", 95: "Province-State",
         101: "Country-PrimaryLocationName", 80: "By-line", 105: "Headline", 110: "Credit", 115: "Source"}


def _read_iptc(img, meta):
    try:
        info = IptcImagePlugin.getiptcinfo(img) or {}
    except Exception:
        return
    for (rec, ds), v in info.items():
        if rec == 2 and ds in _IPTC:
            vals = v if isinstance(v, list) else [v]
            vals = [x.decode("utf-8", "ignore") if isinstance(x, bytes) else str(x) for x in vals]
            meta[f"IPTC:{_IPTC[ds]}"] = vals if len(vals) > 1 or ds == 25 else vals[0]


def _read_image(path: Path) -> dict:
    meta = {}
    img = Image.open(path)
    ex = img.getexif()
    sub, gps = ex.get_ifd(0x8769), ex.get_ifd(0x8825)
    for src in (dict(ex), sub):
        for tag, v in src.items():
            name = _TAGS.get(tag)
            if name and name not in ("MakerNote", "GPSInfo", "ExifOffset") and not (isinstance(v, bytes) and len(v) > 2000):
                meta[f"EXIF:{name}"] = _clean(name, v)
    try:
        if 2 in gps and 4 in gps:
            meta["Composite:GPSLatitude"] = _dms(gps[2], gps.get(1))
            meta["Composite:GPSLongitude"] = _dms(gps[4], gps.get(3))
            meta["EXIF:GPSLatitude"], meta["EXIF:GPSLongitude"] = meta["Composite:GPSLatitude"], meta["Composite:GPSLongitude"]
        if 29 in gps and 7 in gps:  # GPS date + time are UTC
            h, m, s = (int(float(x)) for x in gps[7])
            meta["Composite:GPSDateTime"] = f"{gps[29]} {h:02d}:{m:02d}:{s:02d}Z"
    except Exception:
        pass
    _read_iptc(img, meta)
    _read_xmp(path, meta)
    meta["File:ImageWidth"], meta["File:ImageHeight"] = img.size
    return meta


def _pdf_date(s):
    m = re.match(r"D:(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?([+\-Z])?(\d{2})?'?(\d{2})?", str(s or ""))
    if not m:
        return None
    y, mo, d, h, mi, se, sg, th, tm = m.groups()
    out = f"{y}:{mo or '01'}:{d or '01'} {h or '00'}:{mi or '00'}:{se or '00'}"
    if sg in ("+", "-") and th:
        out += f"{sg}{th}:{tm or '00'}"
    elif sg == "Z":
        out += "Z"
    return out


def _read_pdf(path: Path) -> dict:
    from pypdf import PdfReader
    r = PdfReader(str(path))
    meta = {}
    if r.is_encrypted:
        meta["PDF:Encryption"] = "encrypted"
        return meta
    info = r.metadata or {}
    for k, name in (("/Title", "Title"), ("/Author", "Author"), ("/Producer", "Producer"),
                    ("/Creator", "Creator"), ("/Subject", "Subject"), ("/Keywords", "Keywords")):
        if info.get(k):
            meta[f"PDF:{name}"] = str(info[k])
    for k, name in (("/CreationDate", "CreateDate"), ("/ModDate", "ModifyDate")):
        d = _pdf_date(info.get(k))
        if d:
            meta[f"PDF:{name}"] = d
    meta["PDF:PageCount"] = len(r.pages)
    return meta


def read(path: Path) -> dict:
    ext = Path(path).suffix.lower()
    try:
        if ext == ".pdf":
            return _read_pdf(path)
        return _read_image(path)
    except Exception as e:
        return {"Error": f"fallback reader: {e}"}
