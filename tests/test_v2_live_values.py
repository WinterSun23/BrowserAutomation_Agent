import json
import pytest
from sitegraph.agents.navigator import Navigator
from sitegraph.agents.orchestrator import Orchestrator, _keyword_kind
from sitegraph.blobs import BlobStore
from sitegraph.browser.session import BrowserSession
from sitegraph.capture.segment import build_state
from sitegraph.config import Settings
from sitegraph.embed import HashEmbedder
from sitegraph.events import NullEvents
from sitegraph.llm import FakeLLM
from sitegraph.normalize import domain_of
from sitegraph.ops import Ops
from tests.helpers import site_responder


@pytest.mark.browser
def test_read_values_and_choose_item(site_url):
    with BrowserSession() as s:
        s.goto(site_url + "/products.html")
        obs = build_state(s.snapshot(), domain_of(site_url))
        tpl = next(e for e in obs.elements() if e.is_template and e.role == "link").props()

        def responder(system, user):
            items = json.loads(user.split("Items: ")[1].split("\n")[0])
            cheapest = min(items, key=lambda t: float(t.split("$")[1].split()[0]))
            return json.dumps({"match": cheapest.split(" $")[0]})

        nav = Navigator(None, None, domain_of(site_url), s, Settings(), HashEmbedder(), llm=FakeLLM(responder))
        values = nav.read_values(tpl)
        assert len(values) == 6 and any("$5.00" in v for v in values)
        assert nav.choose_item("open the cheapest product", tpl) == "White Socks"


def test_choose_item_without_a_model_or_a_match_gives_none():
    nav = Navigator(None, None, "x.test", None, Settings(), HashEmbedder(), llm=None)
    assert nav.choose_item("the cheapest", {"is_template": True}) is None


def test_ask_is_a_keyword_kind_but_recommend_and_demo_win():
    assert _keyword_kind("How much is the red hoodie?") == "ask"
    assert _keyword_kind("which product is the cheapest?") == "ask"
    assert _keyword_kind("what should I automate?") == "recommend"
    assert _keyword_kind("what does the demo video show?") == "demo"
    assert _keyword_kind("open the cart") == "automate"            # not a question


@pytest.mark.integration
@pytest.mark.browser
def test_ask_navigates_reads_the_page_and_answers(store, settings, site_url):
    settings.min_target_score = 0.3

    def responder(system, user):
        if "Classify" in user:
            t = user.split("\n", 1)[0].lower()
            if "explore" in t:
                return json.dumps({"kind": "explore"})
            return json.dumps({"kind": "ask", "goal": "open cart", "params": {}})
        if "answer questions about the current web page" in system:
            assert "Question:" in user and "Page text:" in user
            return json.dumps({"answer": "The cart page is open."})
        return site_responder(system, user)

    o = Orchestrator(settings, store, HashEmbedder(), FakeLLM(responder), BlobStore(settings.blob_dir), NullEvents())
    site = o.register_site(site_url + "/index.html")
    ops = Ops(store, settings.uid, site.domain)
    assert o.run_task(ops.create_task("chat", "explore the site"), "explore the site", site)["kind"] == "explore"
    res = o.run_task(ops.create_task("chat", "what is in my cart?"), "what is in my cart?", site)
    assert res["kind"] == "ask" and res["explanation"] == "The cart page is open." and res["status"] == "done", res
