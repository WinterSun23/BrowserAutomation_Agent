"""robots.txt handling: fetched under the bot's own name, an unavailable file (4xx) means no restrictions (RFC 9309)."""
import io
import urllib.error
import urllib.request

import pytest
from sitegraph import policy


def _serve(monkeypatch, text=None, status=200, seen=None):
    def fake_urlopen(req, timeout=None):
        if seen is not None:
            seen.append(req.get_header("User-agent"))
        if status != 200:
            raise urllib.error.HTTPError(req.full_url, status, "x", {}, io.BytesIO(b""))
        return io.BytesIO(text.encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


def test_rules_of_a_served_robots_file_apply(monkeypatch):
    _serve(monkeypatch, "User-agent: *\nDisallow: /private/\n")
    assert policy.robots_allowed("https://rules.example/public/page") is True
    assert policy.robots_allowed("https://rules.example/private/page") is False


def test_robots_is_fetched_under_the_bots_own_name(monkeypatch):
    seen: list = []
    _serve(monkeypatch, "User-agent: *\nDisallow:\n", seen=seen)
    assert policy.robots_allowed("https://agent.example/") is True
    assert seen == ["SiteGraphBot"]            # some sites answer 403 to the default Python agent


@pytest.mark.parametrize("status", [401, 403, 404])
def test_an_unavailable_robots_file_means_everything_is_allowed(monkeypatch, status):
    _serve(monkeypatch, status=status)
    assert policy.robots_allowed(f"https://gone{status}.example/pages/forms/") is True


def test_a_network_error_does_not_block_exploration(monkeypatch):
    def boom(req, timeout=None):
        raise OSError("network down")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    assert policy.robots_allowed("https://offline.example/") is True


def test_local_hosts_are_never_checked():
    assert policy.robots_allowed("http://127.0.0.1:8000/anything") is True
