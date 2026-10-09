"""Generates demo photos WITH realistic metadata so you can demo even without real photos.
Needs exiftool installed. Real photos work much better for CLIP/vision - use these only as a fallback."""
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw

OUT = Path(__file__).parent / "sample_photos"
SETS = [
    # folder, filename, banner text, date, lat, lon, people, keywords, hierarchical, color
    ("Events/2025/Branch Launches/Indiranagar", "IMG_2041.jpg", "AU Small Finance Bank - Indiranagar - Inauguration",
     "2025:07:14 11:32:08", 12.9784, 77.6408, ["Anita Rao", "Vikram Shah"], ["ribbon cutting", "branch", "crowd"], "Events|Branch Launch|Indiranagar", (200, 60, 60)),
    ("Events/2025/Branch Launches/Indiranagar", "IMG_2042.jpg", "Indiranagar Branch - Guests",
     "2025:07:14 11:40:12", 12.9784, 77.6408, ["Anita Rao"], ["guests", "branch"], "Events|Branch Launch|Indiranagar", (210, 90, 70)),
    ("Events/2024/Town Hall", "IMG_1100.jpg", "Annual Town Hall December 2024",
     "2024:12:10 15:05:00", 26.9124, 75.7873, ["Vikram Shah"], ["town hall", "stage", "audience"], "Events|Town Hall", (60, 90, 200)),
    ("Events/2026/Cyber Workshop", "IMG_3001.jpg", "Cybersecurity Workshop",
     "2026-08-20 10:00:00".replace("-", ":", 2), 28.6139, 77.2090, [], ["workshop", "laptops"], "Events|Cybersecurity Workshop", (60, 160, 90)),
]
for folder, name, text, dt, lat, lon, people, kws, hier, col in SETS:
    d = OUT / folder; d.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1200, 800), col); dr = ImageDraw.Draw(img)
    dr.rectangle([60, 300, 1140, 500], fill=(255, 255, 255)); dr.text((90, 380), text, fill=(0, 0, 0))
    p = d / name; img.save(p, "JPEG")
    args = ["exiftool", "-overwrite_original", f"-DateTimeOriginal={dt}", "-OffsetTimeOriginal=+05:30",
            "-Make=Canon", "-Model=EOS R6", f"-GPSLatitude={abs(lat)}", "-GPSLatitudeRef=N", f"-GPSLongitude={abs(lon)}", "-GPSLongitudeRef=E",
            f"-XMP-lr:HierarchicalSubject={hier}"]
    args += [f"-IPTC:Keywords={k}" for k in kws] + [f"-XMP-iptcExt:PersonInImage={n}" for n in people] + [str(p)]
    subprocess.run(args, capture_output=True)
# a WhatsApp-style file with stripped metadata (date only in filename)
d = OUT / "WhatsApp Images"; d.mkdir(parents=True, exist_ok=True)
Image.new("RGB", (800, 600), (120, 120, 120)).save(d / "IMG-20250714-WA0012.jpg", "JPEG")
print("samples written to", OUT)
