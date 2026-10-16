import pytest
from sitegraph.schema import ensure_device_schema, ensure_ops_schema, ensure_profile_schema

pytestmark = pytest.mark.integration


def _indexes(store, key):
    return store.ro_query(key, "CALL db.indexes() YIELD label, properties RETURN label, properties")


def test_device_schema_is_idempotent(store, gkey):
    ensure_device_schema(store, gkey, 384)
    ensure_device_schema(store, gkey, 384)
    rows = _indexes(store, gkey)
    labels = {r[0] for r in rows}
    assert {"State", "Element", "Intent", "Workflow", "Component"} <= labels
    assert any(r[0] == "State" and "emb" in r[1] for r in rows)


def test_ops_and_profile_schema(store, gkey):
    ensure_ops_schema(store, gkey + ":ops")
    ensure_profile_schema(store, gkey + ":profile", 384)
    assert any(r[0] == "Task" for r in _indexes(store, gkey + ":ops"))
    assert any(r[0] == "Memory" and "emb" in r[1] for r in _indexes(store, gkey + ":profile"))
