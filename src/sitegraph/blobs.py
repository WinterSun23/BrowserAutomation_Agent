from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path

from PIL import Image


class BlobStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, ref: str) -> Path:
        return self.root / ref[:2] / ref[2:4] / ref

    def put_bytes(self, data: bytes, ext: str) -> str:
        ref = hashlib.sha256(data).hexdigest() + ext
        p = self.path(ref)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        return ref

    def put_file(self, src: Path | str) -> str:
        src = Path(src)
        return self.put_bytes(src.read_bytes(), "".join(src.suffixes) or ".bin")

    def put_screenshot(self, png: bytes, quality: int = 70) -> str:
        buf = io.BytesIO()
        Image.open(io.BytesIO(png)).convert("RGB").save(buf, "WEBP", quality=quality)
        return self.put_bytes(buf.getvalue(), ".webp")

    def put_json_gz(self, obj) -> str:
        data = gzip.compress(json.dumps(obj, sort_keys=True).encode(), mtime=0)
        return self.put_bytes(data, ".json.gz")

    def read(self, ref: str) -> bytes:
        return self.path(ref).read_bytes()

    def delete(self, ref: str) -> None:
        self.path(ref).unlink(missing_ok=True)
