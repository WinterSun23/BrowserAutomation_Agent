import io
from PIL import Image
from sitegraph.blobs import BlobStore


def _png():
    buf = io.BytesIO()
    Image.new("RGB", (20, 10), (200, 30, 30)).save(buf, "PNG")
    return buf.getvalue()


def test_put_bytes_dedupes(tmp_path):
    b = BlobStore(tmp_path)
    r1 = b.put_bytes(b"hello", ".txt")
    r2 = b.put_bytes(b"hello", ".txt")
    assert r1 == r2 and r1.endswith(".txt") and b.read(r1) == b"hello"


def test_screenshot_is_webp(tmp_path):
    ref = BlobStore(tmp_path).put_screenshot(_png())
    assert ref.endswith(".webp")


def test_json_gz_roundtrip_is_deterministic(tmp_path):
    b = BlobStore(tmp_path)
    assert b.put_json_gz({"a": 1}) == b.put_json_gz({"a": 1})
