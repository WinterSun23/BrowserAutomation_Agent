import pytest
from sitegraph.risk import classify_risk, RISK_COST


@pytest.mark.parametrize("role,name,href,expected", [
    ("link", "products", "http://s.test/products.html", "navigate"),
    ("link", "checkout", "http://s.test/checkout.html", "submit_mutating"),
    ("link", "twitter", "https://twitter.com/x", "external"),
    ("link", "mail us", "mailto:a@b.c", "external"),
    ("button", "add to cart", None, "submit_safe"),
    ("button", "search", None, "submit_safe"),
    ("button", "place order", None, "submit_mutating"),
    ("button", "delete account", None, "destructive"),
    ("searchbox", "search products", None, "input"),
    ("menuitem", "help", "http://s.test/about.html", "navigate"),
])
def test_classify(role, name, href, expected):
    assert classify_risk(role, name, href, "s.test") == expected


def test_costs():
    assert RISK_COST["submit_safe"] == 1 and RISK_COST["submit_mutating"] == 100
