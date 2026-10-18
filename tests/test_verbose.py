import logging

import pytest
from sitegraph.cli import build_parser, setup_logging
from sitegraph.events import ConsoleEvents


def test_console_events_prints_one_line_per_known_event():
    lines = []
    ev = ConsoleEvents(write=lines.append)
    ev.publish("u", {"type": "state", "title": "Cart", "url_template": "/cart.html", "new": True, "agent": "explorer-0"})
    ev.publish("u", {"type": "step", "label": "click 'Cart'", "ok": True, "jpeg": "xxx"})
    ev.publish("u", {"type": "explore_done", "states": 5, "actions": 7, "transitions": 1, "blocked": 0})
    ev.publish("u", {"type": "something_else"})
    assert len(lines) == 3
    assert lines[0].startswith("+ state Cart (/cart.html) [new] by explorer-0")
    assert "jpeg" not in lines[1] and "click 'Cart'" in lines[1]
    assert lines[2].startswith("explore done")


def test_verbose_flags_parse_on_every_command():
    for argv, level in ((["run", "http://x.test/", "explore", "-v"], 1), (["chat", "http://x.test/", "-vv"], 2),
                        (["mcp", "http://x.test/", "-v"], 1)):
        assert build_parser().parse_args(argv).verbose == level
    assert build_parser().parse_args(["run", "http://x.test/", "explore"]).verbose == 0


def test_setup_logging_levels():
    for v, level in ((0, logging.WARNING), (1, logging.INFO), (2, logging.DEBUG)):
        setup_logging(v)
        assert logging.getLogger().level == level
    setup_logging(0)


@pytest.mark.integration
@pytest.mark.browser
def test_exploring_with_console_events_reports_new_states(store, settings, site_url):
    from sitegraph.agents.orchestrator import Orchestrator
    from sitegraph.blobs import BlobStore
    from sitegraph.embed import HashEmbedder
    from sitegraph.llm import OfflineLLM
    from sitegraph.ops import Ops

    lines = []
    settings.navigators_per_site, settings.dynamic_spawn = 1, False
    o = Orchestrator(settings, store, HashEmbedder(), OfflineLLM(), BlobStore(settings.blob_dir),
                     ConsoleEvents(write=lines.append))
    site = o.register_site(site_url + "/index.html")
    o.run_task(Ops(store, settings.uid, site.domain).create_task("chat", "explore"), "explore the site", site,
               kind="explore")
    assert sum(1 for l in lines if l.startswith("+ state") and "[new]" in l) >= 5
    assert any(l.startswith("explore done") for l in lines)
