# Photo Archivist Agent (MVP)

NAS/local folder -> **exiftool metadata first** -> Claude vision (caption/tags/OCR) -> CLIP embeddings -> SQLite index
-> organised *copy* (dry-run + undo) -> FastAPI search + web UI. Originals are never touched.

## Setup (5 min)
```bash
sudo apt install libimage-exiftool-perl      # mac: brew install exiftool | windows: exiftool.org
python -m venv venv && source venv/bin/activate   # windows: venv\Scripts\activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...          # optional: without it vision is skipped, rest still works
export SOURCE_DIR=/path/to/your/photos       # default: ./sample_photos
python make_samples.py                       # optional demo photos with metadata
```

## Demo flow
```bash
python -m archivist.cli dry-run              # 1. coverage report, writes NOTHING
python -m archivist.cli index                # 2. metadata + vision + embeddings -> data/index.db
python -m archivist.cli organise             # 3. plan folders (dry-run)
python -m archivist.cli organise --apply     # 4. copy into data/Organised + sidecar JSON
python -m archivist.cli undo                 # reverse last batch
python -m archivist.cli serve                # 5. http://localhost:8000
```
Flags: `index --no-vision --no-embed --limit 20`. Env: `THRESH_HIGH`, `THRESH_LOW`, `EXCLUDE`, `VISION_MODEL`.
Edit `roles.json` to map "the MD" / "the CFO" to real names.

## Files
| file | job |
|---|---|
| `metadata.py` | exiftool dump + trust-order reconcile (date/place/people/event) + sanity checks |
| `vision.py` | Claude vision: caption, tags, OCR (never names people from faces) |
| `embed.py` | CLIP image+text vectors (local) |
| `indexer.py` | scan, sha256, duplicates (reported only), dry-run stats, write index |
| `organiser.py` | folder score 0.5 vec + 0.3 tags + 0.2 people; >=.8 copy, .6-.79 `_Review`, <.6 new; undo log |
| `search.py` | query -> filters (year/month/person/role/place/type) + vectors + keywords; reason+provenance per hit |
| `app.py` + `static/index.html` | API + frontend |

## Company machine setup (no exiftool / Qwen or Claude)

**Metadata:** if `exiftool` isn't installed the app automatically uses a built-in pure-Python reader
(EXIF + GPS + IPTC + XMP people/keywords + PDF info; verified identical to exiftool on the test set).
Missing vs exiftool: MakerNotes, video, HEIC. If you can, unzip portable exiftool (exiftool.org) anywhere and set
`EXIFTOOL_PATH`. Test the fallback anytime with `FORCE_FALLBACK=1`.

**Vision (pick one):**
```bash
# Claude
export VISION_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-ant-... VISION_MODEL=claude-sonnet-4-5
# Qwen-VL on your own server (data never leaves the company) - Ollama example
export VISION_PROVIDER=openai VISION_BASE_URL=http://<server>:11434/v1 VISION_MODEL=qwen2.5vl:7b
# Qwen via DashScope API: VISION_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1 VISION_API_KEY=... VISION_MODEL=qwen-vl-max
```
Windows PowerShell: `$env:VISION_PROVIDER="openai"` instead of `export`.

**CLIP blocked / no internet?** Download `clip-ViT-B-32` once on another machine, copy the folder over and set
`CLIP_MODEL=C:\path\to\clip-ViT-B-32`. Or run `index --no-embed` (keyword + filter search still works).

**People:** edit `people.json` (`"Sanjay": ["sanjay"]`). A match from filename = confidence 0.6, from a folder name
like `Sanjay Sir/` = 0.4 (folder owner is not necessarily who's in the photo, so it is labelled "unverified").
