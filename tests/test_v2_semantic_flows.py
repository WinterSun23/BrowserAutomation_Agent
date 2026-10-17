import pytest
from sitegraph.embed import HashEmbedder
from sitegraph.llm import FakeLLM
from sitegraph.semantic.distiller import Distiller, semantic_key
from tests.test_v2_semantic import responder, two_devices  # noqa: F401  (module fixture reuse)

pytestmark = [pytest.mark.integration, pytest.mark.browser]


def test_flow_patterns_and_followed_by(store, msettings, two_devices):
    prof, _, _ = two_devices
    stats = Distiller(store, msettings.uid, HashEmbedder(), FakeLLM(responder), msettings, prof).rebuild()
    sk = semantic_key(msettings.uid)
    assert stats["step_patterns"] >= 1 and stats["flow_patterns"] >= 1 and stats["followed_by"] >= 1
    assert store.ro_query(sk, "MATCH (:WorkflowRef)-[:INSTANCE_OF]->(f:FlowPattern)-[:STEP]->(:StepPattern) "
                              "RETURN count(DISTINCT f) >= 1") == [[True]]
    assert store.ro_query(sk, "MATCH (u:UIPattern) WHERE u.pagerank IS NULL RETURN count(u)") == [[0]]
    assert store.ro_query(sk, "MATCH (:LogicalState)-[:INSTANCE_OF]->(:StepPattern) RETURN count(*) > 0") == [[True]]
