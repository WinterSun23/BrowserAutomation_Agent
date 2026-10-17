from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
from pathlib import Path
from typing import Protocol


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def cosine(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


class HashEmbedder:
    """Deterministic bag-of-tokens embedder for tests (no model download)."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for tok in re.findall(r"[a-z0-9]+", (t or "").lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                v[h % self.dim] += 1.0 if (h >> 8) & 1 else -1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


class BgeEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name           # recorded on the Site node: vectors of two models must not be mixed
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self._cache: dict[str, list[float]] = {}

    def embed(self, texts: list[str]) -> list[list[float]]:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            vecs = self.model.encode(missing, normalize_embeddings=True).tolist()
            self._cache.update(zip(missing, vecs))
        return [self._cache[t] for t in texts]


class CachedEmbedder:
    """Persistent embedding cache (SQLite) in front of any embedder: a text is embedded once per model, ever."""

    def __init__(self, inner, path: Path):
        self.inner, self.dim = inner, inner.dim
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock:
            self.db.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v TEXT)")
            self.db.commit()
        self.tag = getattr(inner, "model_name", type(inner).__name__)

    @property
    def model_name(self) -> str:
        """The wrapped model's name: the Site node records which model built a graph, and a cache must not change it."""
        return self.tag

    def _k(self, t: str) -> str:
        return hashlib.sha1(f"{self.tag}\x1f{t}".encode()).hexdigest()

    def embed(self, texts: list[str]) -> list[list[float]]:
        keys = [self._k(t) for t in texts]
        found: dict[str, list[float]] = {}
        with self.lock:
            for k in set(keys):
                row = self.db.execute("SELECT v FROM emb WHERE k=?", (k,)).fetchone()
                if row:
                    found[k] = json.loads(row[0])
        missing = list(dict.fromkeys(t for t, k in zip(texts, keys) if k not in found))
        if missing:
            vecs = self.inner.embed(missing)
            with self.lock:
                for t, v in zip(missing, vecs):
                    found[self._k(t)] = v
                    self.db.execute("INSERT OR REPLACE INTO emb VALUES (?, ?)", (self._k(t), json.dumps(v)))
                self.db.commit()
        return [found[k] for k in keys]
