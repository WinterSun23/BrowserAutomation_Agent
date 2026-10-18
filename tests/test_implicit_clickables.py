import pytest
from sitegraph.agents.navigator import Navigator
from sitegraph.browser.session import BrowserSession
from sitegraph.capture.model import RawNode, RawSnapshot
from sitegraph.capture.segment import build_state
from sitegraph.db import device_key
from sitegraph.embed import HashEmbedder
from sitegraph.normalize import domain_of
from sitegraph.schema import ensure_device_schema
from sitegraph.versions import ALGO_VERSION

pytestmark = [pytest.mark.integration, pytest.mark.browser]

HINTED = {"Attribute handler", "Class button", "Focusable thing", "Pointer cursor"}


def _inferred(snapshot) -> dict[str, RawNode]:
    # the card's two spans have no space between them in the page text: compare ignoring spaces
    return {("Card title Card price" if n.name.replace(" ", "") == "CardtitleCardprice" else n.name): n
            for n in snapshot.nodes if n.inferred}


def test_elements_without_a_role_but_with_a_click_hint_or_listener_are_captured_as_buttons(site_url):
    with BrowserSession() as s:
        s.goto(site_url + "/implicit.html")
        found = _inferred(s.snapshot())
    assert HINTED | {"Script listener", "Card title Card price"} <= set(found)
    assert all(n.role == "button" and n.interactive and n.tag != "a" for n in found.values())
    # never: nothing listens, no name to identify it by, a wrapper around a real link, a page-sized container
    assert set(found) == HINTED | {"Script listener", "Card title Card price"}


def test_without_listener_detection_only_the_cheap_hints_are_used(site_url):
    with BrowserSession(detect_listeners=False) as s:
        s.goto(site_url + "/implicit.html")
        found = _inferred(s.snapshot())
    assert "Script listener" not in found and HINTED <= set(found)


def test_real_roles_are_untouched_and_the_wrapped_link_stays_the_target(site_url):
    with BrowserSession() as s:
        s.goto(site_url + "/implicit.html")
        snap = s.snapshot()
    links = [n for n in snap.nodes if n.role == "link"]
    assert [n.name for n in links] == ["Real link"] and not links[0].inferred


def test_an_inferred_button_is_an_element_that_can_be_stored_and_clicked(store, settings, site_url):
    domain = domain_of(site_url)
    key = device_key(settings.uid, domain, "desktop")
    ensure_device_schema(store, key, 384)
    with BrowserSession() as s:
        nav = Navigator(store, key, domain, s, settings, HashEmbedder())
        s.goto(site_url + "/implicit.html")
        obs = nav.observe()
        els = {e.name_sample: e for e in obs.elements()}
        assert els["Script listener"].role == "button" and els["Script listener"].inferred is True
        assert els["Script listener"].risk == "submit_safe"
        nav.write_state(obs)
        stored = store.ro_query(key, "MATCH (x:Element) WHERE x.inferred = true RETURN count(x)")[0][0]
        assert stored == len(HINTED) + 2
        located = nav.resolver.resolve(els["Script listener"].props())
        assert located is not None
        located.locator.click()
        assert s.page.inner_text("#out") == "listener clicked"


def test_the_identity_rules_changed_so_existing_graphs_are_rehashed():
    assert ALGO_VERSION >= 2


def test_snapshots_stored_before_the_change_still_load():
    old = {"url": "http://x.test/", "title": "t", "nodes": [{
        "idx": 0, "role": "link", "name": "Home", "tag": "a", "href": "http://x.test/", "input_type": None,
        "parent": None, "depth": 0, "interactive": True, "states": {}, "bbox": [0, 0, 1, 1], "css": "a", "gcss": "a",
        "testid": None, "attrs": {}}]}
    snap = RawSnapshot.from_dict(old)
    assert snap.nodes[0].inferred is False
    assert build_state(snap, "x.test", 4).elements()[0].inferred is False
