"""Cheap offline colour estimate for the CLOTHING area (lower part of a portrait). Pillow + numpy only.
It is a HEURISTIC: it can be fooled by backgrounds/lighting, so it is only stored as a tag worded
'est. <colour> clothing' and used as a ranking hint. Vision (Claude/Qwen) is the proper source."""
import numpy as np
from PIL import Image, ImageOps


def clothing_colours(path, min_share=0.25):
    try:
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        img.thumbnail((160, 160))
        w, h = img.size
        region = img.crop((0, int(h * 0.62), w, h))  # face is above this line in typical portraits
        hsv = np.asarray(region.convert("HSV"), dtype=np.float32) / 255.0
    except Exception:
        return []
    H, S, V = hsv[..., 0] * 360.0, hsv[..., 1], hsv[..., 2]
    total = H.size
    skin = (H >= 5) & (H <= 38) & (S >= 0.18) & (S <= 0.68) & (V > 0.68)  # ignore bright faces/hands
    black = V < 0.18
    neutral = (S < 0.15) & ~black
    chroma = ~skin & ~black & ~neutral
    fam = {
        "brown": chroma & ((H >= 348) | (H < 48)) & (S >= 0.28) & (V <= 0.72) & ~((S >= 0.5) & ((H < 8) | (H >= 345))),
        "orange": chroma & (H >= 8) & (H < 40) & (V > 0.62) & (S > 0.5),
        "red": chroma & ((H < 8) | (H >= 345)) & (S >= 0.5),
        "yellow": chroma & (H >= 48) & (H < 70),
        "green": chroma & (H >= 70) & (H < 170),
        "blue": chroma & (H >= 170) & (H < 260),
        "purple": chroma & (H >= 260) & (H < 300),
        "pink": chroma & (H >= 300) & (H < 345),
        "black": black,
    }
    shares = {k: float(m.sum()) / total for k, m in fam.items()}
    return [k for k, v in sorted(shares.items(), key=lambda x: -x[1]) if v >= min_share][:2]
