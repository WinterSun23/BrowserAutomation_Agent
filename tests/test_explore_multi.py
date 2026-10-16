import threading

import pytest
from sitegraph.agents.navigator import Navigator
from sitegraph.browser.session import BrowserSession
from sitegraph.db import device_key
from sitegraph.embed import HashEmbedder
from sitegraph.normalize import domain_of
from sitegraph.ops import Ops
from sitegraph.policy import ExplorationPolicy
from sitegraph.schema import ensure_device_schema

pytestmark = [pytest.mark.integration, pytest.mark.browser]


def test_two_navigators_share_one_site_without_repeating_work(store, settings, site_url):
    domain = domain_of(site_url)
    key = device_key(settings.uid, domain, "desktop")
    ensure_device_schema(store, key, 384)
    scope, reports, errors = "scope-multi", {}, []

    def agent(n: int) -> None:
        try:
            with BrowserSession() as s:
                nav = Navigator(store, key, domain, s, settings, HashEmbedder(), agent_id=f"navigator-desktop-{n}")
                reports[n] = nav.explore(site_url + "/index.html", ExplorationPolicy(max_depth=2), scope=scope)
        except Exception as exc:  # surfaced by the assertion below
            errors.append(repr(exc))

    threads = [threading.Thread(target=agent, args=(n,)) for n in (1, 2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors

    ops = Ops(store, settings.uid, domain)
    stats = ops.item_stats(scope)
    assert "open" not in stats                                   # everything was worked off
    assert stats["done"]["max_attempts"] == 1                    # no item was processed twice
    done_by = {r[0] for r in ops.q("MATCH (w:WorkItem {scope:$s}) RETURN DISTINCT w.done_by", s=scope)}
    assert done_by == {"navigator-desktop-1", "navigator-desktop-2"}   # the work really was shared
    assert store.ro_query(key, "MATCH (s:State) RETURN count(s)")[0][0] >= 6
    assert store.ro_query(key, "MATCH (e:Element) WHERE e.risk_cost > 1 AND e.tried = true RETURN count(e)") == [[0]]
    assert store.ro_query(ops.key, "MATCH (c:Claim) RETURN count(c)") == [[0]]   # all leases released
