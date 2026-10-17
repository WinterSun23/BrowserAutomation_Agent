import io
import pytest
from PIL import Image
from sitegraph.agents.navigator import Navigator
from sitegraph.blobs import BlobStore
from sitegraph.browser.session import BrowserSession
from sitegraph.db import device_key
from sitegraph.embed import HashEmbedder
from sitegraph.graph.queries import RouteStep
from sitegraph.normalize import domain_of
from sitegraph.policy import ExplorationPolicy
from sitegraph.schema import ensure_device_schema
from sitegraph.visual import crop_png, dhash, draw_marks, hamming_hex, side_by_side


def _png(color, size=(64, 48)):
    buf = io.BytesIO()
    img = Image.new("RGB", size, color)
    img.paste((0, 0, 0), (0, 0, size[0] // 2, size[1] // 2))
    img.save(buf, "PNG")
    return buf.getvalue()


def test_dhash_stable_and_sensitive():
    a, b = _png((255, 255, 255)), _png((255, 255, 255))
    assert dhash(a) == dhash(b) and len(dhash(a)) == 16
    flipped = Image.open(io.BytesIO(a)).transpose(Image.FLIP_LEFT_RIGHT)
    buf = io.BytesIO(); flipped.save(buf, "PNG")
    assert hamming_hex(dhash(a), dhash(buf.getvalue())) > 6


def test_crop_marks_and_side_by_side():
    png = _png((200, 200, 200))
    crop = Image.open(io.BytesIO(crop_png(png, [10, 10, 20, 10], pad=0)))
    assert crop.size == (20, 10)
    assert Image.open(io.BytesIO(draw_marks(png, [[5, 20, 10, 10]]))).size == (64, 48)
    assert Image.open(io.BytesIO(side_by_side(png, png))).size == (64 * 2 + 20, 48)


@pytest.mark.integration
@pytest.mark.browser
def test_explore_stores_phash_crops_and_failure_evidence(store, settings, site_url, tmp_path):
    domain = domain_of(site_url)
    key = device_key(settings.uid, domain, "desktop")
    ensure_device_schema(store, key, 384)
    blobs = BlobStore(tmp_path / "blobs")
    with BrowserSession() as s:
        nav = Navigator(store, key, domain, s, settings, HashEmbedder(), blobs=blobs)
        nav.explore(site_url + "/index.html", ExplorationPolicy(max_depth=1))
        assert store.ro_query(key, "MATCH (s:State) WHERE s.phash IS NULL RETURN count(s)") == [[0]]
        assert store.ro_query(key, "MATCH (:Component {kind:'menu'})-[:HAS_CSTATE]->(c:ComponentState {name:'open'}) "
                                   "RETURN c.screenshot_ref IS NOT NULL") == [[True]]
        run = nav.activity.start_run("execute", "t")
        out = nav.execute_step(RouteStep(action_gid="ac:none", label="x", el_gid="el:missing", src="st:none",
                                         dst="st:none"), run, 0, {})
        assert not out.ok
        assert store.ro_query(key, "MATCH (st:Step {gid:$g}) RETURN st.screenshot_ref IS NOT NULL",
                              {"g": out.step_gid}) == [[True]]
