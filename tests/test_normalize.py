import pytest
from sitegraph.normalize import url_template, domain_of, normalize_text, sample_text, var_name


@pytest.mark.parametrize("url,expected", [
    ("https://x.com/product/123?ref=abc&a=1", "/product/:id?a&ref"),
    ("https://x.com/catalogue/a-light-in-the-attic_1000/index.html", "/catalogue/:id/index.html"),
    ("https://x.com/catalogue/category/books/travel_2/index.html", "/catalogue/category/books/:id/index.html"),
    ("https://x.com/about.html", "/about.html"),
    ("https://x.com/", "/"),
    ("https://x.com", "/"),
    ("http://127.0.0.1:8000/product.html?id=4", "/product.html?id"),
    ("https://x.com/u/0f8fad5b-d9cb-469f-a165-70867728950e", "/u/:id"),
    ("https://x.com/page-2.html", "/:id.html"),
    ("https://x.com/a#frag", "/a"),
])
def test_url_template(url, expected):
    assert url_template(url) == expected


def test_domain_of():
    assert domain_of("https://WWW.Example.com:443/a") == "www.example.com:443"
    assert domain_of("http://127.0.0.1:5000/x") == "127.0.0.1:5000"


@pytest.mark.parametrize("raw,expected", [
    ("Cart (3)", "cart (#)"),
    ("Price: £51.77", "price: <money>"),
    ("  Add   to\n cart ", "add to cart"),
    ("A Light in the Attic and more words", "a light in the …"),
    (None, ""),
])
def test_normalize_text(raw, expected):
    assert normalize_text(raw) == expected


def test_sample_text_truncates_and_collapses():
    assert sample_text("a  b\n c") == "a b c"
    assert len(sample_text("x" * 300)) == 200


def test_var_name():
    assert var_name("search products", "q") == "q"
    assert var_name("e-mail address") == "e_mail_address"
    assert var_name("") == "value"
