"""Vision step - runs AFTER metadata. Claude describes the scene, reads text (OCR), suggests tags.
It never names people from faces: names only come from metadata (spec: vision guess is lowest trust)."""
import base64
import io
import json
import re

from PIL import Image, ImageOps

from . import config

PROMPT = """You are indexing a photo archive for a bank. Look at the image and reply with JSON ONLY:
{"caption": "<one factual sentence>",
 "tags": ["5-10 short lowercase tags: objects, setting, activity"],
 "attributes": ["short visual descriptors of each visible person's clothing/accessories/activity, e.g. 'brown blazer', 'black-rimmed glasses', 'speaking at podium', 'blue checked shirt'"],
 "ocr_text": "<all legible text/signage/banners, or empty string>",
 "event_guess": "<short event name if obvious (e.g. branch launch, workshop), else null>",
 "setting": "indoor|outdoor|unknown",
 "pii_visible": <true if ID cards, account numbers, phone numbers or customer documents are legible>}
Rules: do NOT identify real people by their face. Describe only what is visible. If unsure use null or []."""

_client = None


def _prep(path) -> str:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # respect Orientation before any vision call
    img = img.convert("RGB")
    img.thumbnail((1024, 1024))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode()


def _call_anthropic(b64: str) -> str:
    global _client
    import anthropic
    _client = _client or anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    msg = _client.messages.create(
        model=config.VISION_MODEL, max_tokens=600,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
            {"type": "text", "text": PROMPT}]}])
    return "".join(b.text for b in msg.content if b.type == "text")


def _call_openai_compat(b64: str) -> str:
    """Ollama / vLLM / LM Studio / DashScope (Qwen-VL) - anything speaking /v1/chat/completions. stdlib only."""
    import urllib.request
    body = {"model": config.VISION_MODEL, "max_tokens": 600, "temperature": 0,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}]}]}
    req = urllib.request.Request(config.VISION_BASE_URL.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + config.VISION_API_KEY})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def analyse(path) -> "dict | None":
    if not config.USE_VISION:
        return None
    try:
        b64 = _prep(path)
        text = _call_anthropic(b64) if config.VISION_PROVIDER == "anthropic" else _call_openai_compat(b64)
        m = re.search(r"\{.*\}", text, re.S)  # small models wrap JSON in prose/markdown - extract it
        data = json.loads(m.group(0)) if m else None
        if data:
            data["tags"] = [str(t) for t in (data.get("tags") or [])]
            data["attributes"] = [str(t) for t in (data.get("attributes") or [])]
        return data
    except Exception as e:
        print(f"   [vision failed: {e}]")
        return None
