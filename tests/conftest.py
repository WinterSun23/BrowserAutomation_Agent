import functools
import http.server
import os
import threading
import uuid
from pathlib import Path

import pytest

# Tests never read the developer's .env (it may point at a real model); subprocesses inherit this too.
# LiteLLM loads .env into os.environ on import unless LITELLM_MODE is set to something other than DEV.
os.environ["SITEGRAPH_ENV_FILE"] = ""
os.environ["LITELLM_MODE"] = "PRODUCTION"

from sitegraph.config import Settings
from sitegraph.db import GraphStore

SITE_DIR = Path(__file__).parent / "fixtures" / "site"


def _falkor_up() -> bool:
    try:
        return bool(GraphStore().redis.ping())
    except Exception:
        return False


@pytest.fixture(scope="session")
def store():
    if not _falkor_up():
        pytest.skip("FalkorDB not running on localhost:6379 (run: docker compose up -d)")
    return GraphStore()


@pytest.fixture
def gkey(store):
    key = f"test:{uuid.uuid4().hex[:10]}"
    yield key
    for k in store.list_graphs(key):
        store.delete(k)


@pytest.fixture
def settings(tmp_path, store):
    uid = f"t{uuid.uuid4().hex[:8]}"
    s = Settings(uid=uid, data_dir=tmp_path / "data")
    yield s
    for k in store.list_graphs(f"u:{uid}:"):
        store.delete(k)


@pytest.fixture(scope="module")
def msettings(tmp_path_factory, store):
    """Module-scoped settings for tests that build one explored graph and reuse it."""
    uid = f"t{uuid.uuid4().hex[:8]}"
    s = Settings(uid=uid, data_dir=tmp_path_factory.mktemp("data"), min_target_score=0.3)
    yield s
    for k in store.list_graphs(f"u:{uid}:"):
        store.delete(k)


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@pytest.fixture(scope="session")
def site_url():
    handler = functools.partial(_QuietHandler, directory=str(SITE_DIR))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
