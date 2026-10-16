import pytest
from sitegraph.browser.session import BrowserSession
from sitegraph.capture.segment import build_state, jaccard
from sitegraph.normalize import domain_of
from tests.helpers import node, page


def test_jaccard():
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard({"a"}, {"b"}) == 0.0
    assert jaccard(set(), set()) == 1.0


def test_links_identified_by_destination_not_text():
    a = page("http://s.test/p.html", node(0, "navigation", "Crumbs", gcss="nav"),
             node(1, "link", "Travel", parent=0, depth=1, href="http://s.test/cat/travel_2/index.html"))
    b = page("http://s.test/p.html", node(0, "navigation", "Crumbs", gcss="nav"),
             node(1, "link", "Mystery", parent=0, depth=1, href="http://s.test/cat/mystery_3/index.html"))
    assert a.state_gid == b.state_gid
    assert a.elements()[0].gid == b.elements()[0].gid
    assert a.elements()[0].name_sample == "Travel"


def test_template_detection_requires_same_destination():
    items = [node(0, "list", "Products", gcss="ul.grid")]
    for i in range(5):
        li = 1 + i * 3
        items += [node(li, "listitem", f"Item {i}", parent=0, depth=1, gcss="ul.grid > li"),
                  node(li + 1, "link", f"Item {i}", parent=li, depth=2, href=f"http://s.test/product.html?id={i}",
                       gcss="ul.grid > li > a"),
                  node(li + 2, "button", "Add to cart", parent=li, depth=2, gcss="ul.grid > li > button")]
    obs = page("http://s.test/products.html", *items)
    els = obs.elements()
    assert len(els) == 2 and all(e.is_template and e.count == 5 for e in els)
    link = next(e for e in els if e.role == "link")
    assert link.name_norm == "<item>" and link.item_key_fields == ["link"]
    assert link.locators[0] == "item=ul.grid > li||a"
    assert link.sample_items[0] == "Item 0"

    menu = [node(0, "list", "Menu", gcss="ul.menu")]
    for i, target in enumerate(["home", "about", "help", "blog"]):
        li = 1 + i * 2
        menu += [node(li, "listitem", target, parent=0, depth=1, gcss="ul.menu > li"),
                 node(li + 1, "link", target, parent=li, depth=2, href=f"http://s.test/{target}.html", gcss="ul.menu > li > a")]
    assert not any(e.is_template for e in page("http://s.test/x.html", *menu).elements())


def test_blocking_dialog_changes_state_and_ephemeral_menu_does_not():
    base = [node(0, "banner", gcss="header"), node(1, "button", "Account", parent=0, depth=1)]
    plain = page("http://s.test/a.html", *base)
    with_menu = page("http://s.test/a.html", *base, node(2, "menu", gcss="div.menu"),
                     node(3, "menuitem", "Help", parent=2, depth=1, href="http://s.test/help.html"))
    with_dialog = page("http://s.test/a.html", *base, node(2, "dialog", "Added", modal=True, gcss="dialog"),
                       node(3, "button", "Close", parent=2, depth=1))
    assert with_menu.state_gid == plain.state_gid
    assert with_dialog.state_gid != plain.state_gid and with_dialog.blocking_sig


@pytest.mark.browser
def test_fixture_pages_segment_as_expected(site_url):
    with BrowserSession() as s:
        s.goto(site_url + "/index.html")
        idx = build_state(s.snapshot(), domain_of(site_url))
        s.goto(site_url + "/products.html")
        prod = build_state(s.snapshot(), domain_of(site_url))
        banner_i = next(c for c in idx.components if c.kind == "banner")
        banner_p = next(c for c in prod.components if c.kind == "banner")
        assert banner_i.gid == banner_p.gid
        tpl = [e for e in prod.elements() if e.is_template]
        assert {e.role for e in tpl} == {"link", "button"} and all(e.count == 6 for e in tpl)
        checkout_free = [e for e in idx.elements() if e.role == "searchbox"]
        assert checkout_free and checkout_free[0].risk == "input" and checkout_free[0].in_form


@pytest.mark.browser
def test_product_pages_share_state(site_url):
    with BrowserSession() as s:
        s.goto(site_url + "/product.html?id=1")
        a = build_state(s.snapshot(), domain_of(site_url))
        s.goto(site_url + "/product.html?id=4")
        b = build_state(s.snapshot(), domain_of(site_url))
        assert a.state_gid == b.state_gid and a.url_template == "/product.html?id"
