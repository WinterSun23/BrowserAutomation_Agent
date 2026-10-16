import pytest
from sitegraph.browser.locate import ElementResolver, score_candidate
from sitegraph.browser.session import BrowserSession
from sitegraph.capture.segment import build_state
from sitegraph.embed import HashEmbedder
from sitegraph.normalize import domain_of


def test_score_prefers_same_role_name_and_path():
    fp = {"role": "button", "name": "Account", "attrs": {"id": "account-btn"}, "gcss": "header > div > button",
          "bbox": [100, 10, 60, 20]}
    same = {"role": "button", "name": "My account", "attrs": {"id": "acct"}, "gcss": "header > div > button",
            "bbox": [100, 10, 70, 20]}
    other = {"role": "button", "name": "Search", "attrs": {"id": None}, "gcss": "main > form > button",
             "bbox": [500, 300, 60, 20]}
    assert score_candidate(fp, same) > 0.75 > score_candidate(fp, other)


def _el(obs, role, name):
    return next(e for e in obs.elements() if e.role == role and e.name_sample == name).props()


@pytest.mark.browser
def test_primary_locator_used_when_valid(site_url):
    with BrowserSession() as s:
        s.goto(site_url + "/index.html")
        el = _el(build_state(s.snapshot(), domain_of(site_url)), "button", "Account")
        found = ElementResolver(s, HashEmbedder()).resolve(el)
        assert found and found.method == "locator:0"


@pytest.mark.browser
def test_fingerprint_heals_renamed_button(site_url):
    with BrowserSession() as s:
        s.goto(site_url + "/index.html")
        el = _el(build_state(s.snapshot(), domain_of(site_url)), "button", "Account")
        s.page.evaluate("() => { const b = document.getElementById('account-btn'); b.id = 'acct'; b.textContent = 'My account'; }")
        found = ElementResolver(s, HashEmbedder()).resolve(el)
        assert found is not None and found.method == "fingerprint"
        assert found.locator.inner_text() == "My account"


@pytest.mark.browser
def test_template_item_match(site_url):
    with BrowserSession() as s:
        s.goto(site_url + "/products.html")
        obs = build_state(s.snapshot(), domain_of(site_url))
        link = next(e for e in obs.elements() if e.is_template and e.role == "link").props()
        found = ElementResolver(s, HashEmbedder()).resolve(link, item_match="Green Cap")
        assert found and found.locator.inner_text() == "Green Cap"
