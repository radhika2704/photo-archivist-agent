"""CLIP embeddings (local, offline after first model download). Image and text share one vector space,
so a text query like 'ribbon cutting at bank branch' can be compared directly with image vectors."""
import numpy as np
from PIL import Image, ImageOps

from . import config

_model = None
_failed = False


def model():
    global _model, _failed
    if _failed:
        return None
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(config.CLIP_MODEL)
        except Exception as e:
            import sys; print(f"[embeddings disabled: {e}]", file=sys.stderr)
            _failed = True
            return None
    return _model


def available() -> bool:
    return model() is not None


def embed_image(path):
    m = model()
    if m is None:
        return None
    try:
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        return np.asarray(m.encode(img, normalize_embeddings=True), dtype=np.float32)
    except Exception:
        return None


def embed_text(text: str):
    m = model()
    if m is None or not text.strip():
        return None
    return np.asarray(m.encode(text, normalize_embeddings=True), dtype=np.float32)


def to_blob(v):
    return None if v is None else v.astype(np.float32).tobytes()


def from_blob(b):
    return None if b is None else np.frombuffer(b, dtype=np.float32)
