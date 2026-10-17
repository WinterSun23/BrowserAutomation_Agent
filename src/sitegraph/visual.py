"""Image helpers for screenshots: perceptual hash (dHash), crops, set-of-marks, highlights, side-by-side diffs."""
from __future__ import annotations

import io

from PIL import Image, ImageDraw


def to_png(data: bytes) -> bytes:
    buf = io.BytesIO()
    Image.open(io.BytesIO(data)).convert("RGB").save(buf, "PNG")
    return buf.getvalue()


def dhash(png: bytes, size: int = 8) -> str:
    img = Image.open(io.BytesIO(png)).convert("L").resize((size + 1, size), Image.LANCZOS)
    px = list(img.tobytes())  # greyscale: one byte per pixel (getdata is deprecated in newer Pillow)
    bits = 0
    for row in range(size):
        for col in range(size):
            left, right = px[row * (size + 1) + col], px[row * (size + 1) + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return f"{bits:0{size * size // 4}x}"


def hamming_hex(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def crop_png(png: bytes, bbox, pad: int = 8) -> bytes:
    img = Image.open(io.BytesIO(png))
    x, y, w, h = bbox
    box = (max(0, int(x - pad)), max(0, int(y - pad)), min(img.width, int(x + w + pad)), min(img.height, int(y + h + pad)))
    if box[2] <= box[0] or box[3] <= box[1]:
        return png
    buf = io.BytesIO()
    img.crop(box).save(buf, "PNG")
    return buf.getvalue()


def draw_marks(png: bytes, boxes) -> bytes:
    img = Image.open(io.BytesIO(png)).convert("RGB")
    d = ImageDraw.Draw(img)
    for i, (x, y, w, h) in enumerate(boxes):
        d.rectangle([x, y, x + w, y + h], outline=(255, 61, 113), width=3)
        d.rectangle([x, max(0, y - 16), x + 24, y], fill=(255, 61, 113))
        d.text((x + 4, max(0, y - 15)), str(i), fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def draw_highlight(png: bytes, bbox) -> bytes:
    img = Image.open(io.BytesIO(png)).convert("RGB")
    x, y, w, h = bbox
    ImageDraw.Draw(img).rectangle([x - 4, y - 4, x + w + 4, y + h + 4], outline=(255, 61, 113), width=4)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def side_by_side(left: bytes, right: bytes) -> bytes:
    a = Image.open(io.BytesIO(left)).convert("RGB")
    b = Image.open(io.BytesIO(right)).convert("RGB")
    out = Image.new("RGB", (a.width + b.width + 20, max(a.height, b.height)), (255, 255, 255))
    out.paste(a, (0, 0))
    out.paste(b, (a.width + 20, 0))
    buf = io.BytesIO()
    out.save(buf, "PNG")
    return buf.getvalue()
