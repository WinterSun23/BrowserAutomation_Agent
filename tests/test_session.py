import pytest
from sitegraph.browser.session import BrowserSession

pytestmark = pytest.mark.browser


def test_extracts_roles_names_and_hides_closed_menu(site_url):
    with BrowserSession() as s:
        s.goto(site_url + "/index.html")
        snap = s.snapshot()
        pairs = {(n.role, n.name) for n in snap.nodes}
        assert ("link", "Products") in pairs
        assert ("button", "Account") in pairs
        assert ("searchbox", "Search products") in pairs
        assert any(n.role == "navigation" for n in snap.nodes)
        assert not any(n.role == "menuitem" for n in snap.nodes)
        s.page.get_by_role("button", name="Account").click()
        s.settle()
        assert any(n.role == "menuitem" and n.name == "Help" for n in s.snapshot().nodes)


def test_modal_dialog_flagged(site_url):
    with BrowserSession() as s:
        s.goto(site_url + "/products.html")
        s.page.get_by_role("button", name="Add to cart").first.click()
        s.settle()
        dialogs = [n for n in s.snapshot().nodes if n.role == "dialog"]
        assert dialogs and dialogs[0].states["modal"] == "true"


def test_screenshot_and_frame(site_url):
    with BrowserSession() as s:
        s.goto(site_url + "/index.html")
        assert s.screenshot_png()[:4] == b"\x89PNG"
        assert s.frame_jpeg()[:2] == b"\xff\xd8"
