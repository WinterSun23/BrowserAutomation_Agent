"""The page's own complaints (invalid fields, alerts) are detected so a demo never claims a rejected submit worked."""
from sitegraph.browser.problems import page_problems
from sitegraph.browser.session import BrowserSession

FORM = """<form><label>Departure <input id=d type=datetime-local required></label>
<button type=submit>Post</button></form><div role=alert id=a style="display:none">Seat count is too high</div>"""


def test_invalid_field_and_alert_are_reported_and_a_valid_page_is_clean():
    with BrowserSession(headless=True) as s:
        s.page.set_content(FORM)
        assert any("Departure" in p for p in page_problems(s.page))
        s.page.fill("#d", "2026-10-20T20:00")
        assert page_problems(s.page) == []
        s.page.evaluate("document.getElementById('a').style.display='block'")
        assert page_problems(s.page) == ["Seat count is too high"]
