import pytest
from sitegraph.agents.curator import Curator
from sitegraph.agents.navigator import Navigator
from sitegraph.agents.workflows import load_workflow_steps, save_workflow_from_run
from sitegraph.browser.session import BrowserSession
from sitegraph.db import device_key
from sitegraph.embed import HashEmbedder
from sitegraph.llm import FakeLLM
from sitegraph.normalize import domain_of
from sitegraph.policy import ExplorationPolicy
from sitegraph.schema import ensure_device_schema
from tests.helpers import site_responder

pytestmark = [pytest.mark.integration, pytest.mark.browser]

RENAME_CART = """document.addEventListener('DOMContentLoaded', () => setTimeout(() => {
  const a = document.getElementById('cart-link');
  if (a) { a.id = 'basket-link'; a.firstChild.textContent = 'Basket ('; }
}, 0));"""


@pytest.fixture(scope="module")
def world(store, msettings, site_url):
    settings = msettings
    domain = domain_of(site_url)
    key = device_key(settings.uid, domain, "desktop")
    ensure_device_schema(store, key, 384)
    with BrowserSession() as s:
        Navigator(store, key, domain, s, settings, HashEmbedder()).explore(site_url + "/index.html",
                                                                           ExplorationPolicy(max_depth=2))
    Curator(store, key, settings, HashEmbedder(), FakeLLM(site_responder)).run_all(1)
    return store, settings, key, domain


def _nav(world, s):
    store, settings, key, domain = world
    return Navigator(store, key, domain, s, settings, HashEmbedder())


def test_execute_goal_reaches_cart(world, site_url):
    store, _, key, _ = world
    with BrowserSession() as s:
        res = _nav(world, s).execute_goal("open cart")
        assert res.status == "done", res.explanation
        assert s.url.endswith("/cart.html")
        assert res.route is not None and "reliability" in res.explanation


def test_execute_heals_renamed_cart_link(world, site_url):
    store, _, key, _ = world
    with BrowserSession() as s:
        s.context.add_init_script(RENAME_CART)
        res = _nav(world, s).execute_goal("open cart")
        assert res.status == "done" and res.heals >= 1
        assert s.url.endswith("/cart.html")
    rows = store.ro_query(key, "MATCH (o:Observation {kind:'heal'}) RETURN count(o) > 0")
    assert rows == [[True]]


def test_no_target_for_unknown_goal(world):
    with BrowserSession() as s:
        res = _nav(world, s).execute_goal("zzqx unknown flux capacitor")
        assert res.status == "no_target"


def test_no_route_when_only_risky_path(world):
    store, settings, key, _ = world
    v = HashEmbedder().embed(["open checkout"])[0]
    store.query(key, """MATCH (c:State)<-[:HAS_STATE]-(:Page {url_template:'/cart.html'})
                        CREATE (x:State {gid:'st:fakecheckout', summary:'checkout'}),
                               (c)-[:ACTION {gid:'ac:fake', label:'checkout', el_gid:'el:none', cost:1.0,
                                             risk_cost:100, reliability:0.9, pre_cstates:[], requires_inputs:[]}]->(x),
                               (i:Intent {gid:'in:fakecheckout', text:'open checkout', emb: vecf32($v)}),
                               (i)-[:FULFILLED_BY {score:1.0}]->(x)""", {"v": v})
    with BrowserSession() as s:
        assert _nav(world, s).execute_goal("open checkout").status == "no_route"


def test_save_and_replay_workflow(world):
    store, _, key, _ = world
    with BrowserSession() as s:
        res = _nav(world, s).execute_goal("open cart")
    wf = save_workflow_from_run(store, key, HashEmbedder(), res.run_gid, "Go to cart", "open my shopping cart")
    steps = load_workflow_steps(store, key, wf)
    assert steps and steps[0]["kind"] == "ACTION"
    with BrowserSession() as s:
        res2 = _nav(world, s).execute_workflow(wf)
        assert res2.status == "done" and s.url.endswith("/cart.html")
    assert store.ro_query(key, "MATCH (w:Workflow {gid:$w}) RETURN w.run_count", {"w": wf}) == [[1]]


def test_close_targets_go_to_the_planners_choice(world, monkeypatch):
    """Two targets within choose_margin: planner.choose_target (monkeypatched here) picks the one to drive to."""
    import sys
    import types

    import sitegraph.agents.navigator as navmod
    from sitegraph.graph.queries import Target

    store, settings, key, _ = world
    cart = store.ro_query(key, "MATCH (:Page {url_template:'/cart.html'})-[:HAS_STATE]->(s:State) RETURN s.gid LIMIT 1")[0][0]
    about = store.ro_query(key, "MATCH (:Page {url_template:'/about.html'})-[:HAS_STATE]->(s:State) RETURN s.gid LIMIT 1")[0][0]
    close = [Target("State", about, "about", 0.81), Target("State", cart, "cart", 0.80)]
    monkeypatch.setattr(navmod, "find_targets", lambda *a, **k: list(close))
    seen = {}

    def choose_target(llm, request, candidates, store=None, key=None):
        seen.update(request=request, n=len(candidates))
        return candidates[1]

    monkeypatch.setitem(sys.modules, "sitegraph.agents.planner", types.SimpleNamespace(choose_target=choose_target))
    with BrowserSession() as s:
        nav = Navigator(store, key, world[3], s, settings, HashEmbedder(), llm=FakeLLM())
        res = nav.execute_goal("open the cart please")
        assert res.status == "done" and s.url.endswith("/cart.html")
    assert seen == {"request": "open the cart please", "n": 2}
    # a clear winner (gap above the margin) needs no model
    close[1] = Target("State", cart, "cart", 0.5)
    seen.clear()
    with BrowserSession() as s:
        Navigator(store, key, world[3], s, settings, HashEmbedder(), llm=FakeLLM()).execute_goal("about")
    assert seen == {}
