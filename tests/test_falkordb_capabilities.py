import math
import pytest

pytestmark = pytest.mark.integration


def test_merge_coalesce_upsert(store, gkey):
    for _ in range(2):
        store.query(gkey, "MERGE (n:N {gid:'a'}) SET n.first = coalesce(n.first, 5), n.c = coalesce(n.c, 0) + 1")
    assert store.ro_query(gkey, "MATCH (n:N {gid:'a'}) RETURN n.first, n.c") == [[5, 2]]


def test_set_and_remove_label(store, gkey):
    store.query(gkey, "CREATE (:S {gid:'x'})")
    store.query(gkey, "MATCH (s:S {gid:'x'}) SET s:Frontier")
    assert store.ro_query(gkey, "MATCH (s:S:Frontier) RETURN count(s)") == [[1]]
    store.query(gkey, "MATCH (s:S {gid:'x'}) REMOVE s:Frontier")
    assert store.ro_query(gkey, "MATCH (s:S:Frontier) RETURN count(s)") == [[0]]


def test_unwind_map_params_and_list_ops(store, gkey):
    store.query(gkey, "UNWIND $rows AS r MERGE (n:M {gid:r.g}) SET n.tags = r.t", {"rows": [{"g": "1", "t": ["a", "b"]}]})
    rows = store.ro_query(gkey, "MATCH (n:M) RETURN [x IN n.tags WHERE x <> 'a'], n.tags[0..1], labels(n)")
    assert rows == [[["b"], ["a"], ["M"]]]


def test_log_and_tofloat(store, gkey):
    v = store.query(gkey, "RETURN 1.0 + 2.0 * (0.0 - log(toFloat(3) / 4))")[0][0]
    assert math.isclose(v, 1 + 2 * -math.log(0.75), rel_tol=1e-6)


def test_vector_index_and_query(store, gkey):
    store.query(gkey, "CREATE VECTOR INDEX FOR (n:V) ON (n.emb) OPTIONS {dimension:3, similarityFunction:'cosine'}")
    store.query(gkey, "CREATE (:V {gid:'a', emb: vecf32([1.0, 0.0, 0.0])}), (:V {gid:'b', emb: vecf32([0.0, 1.0, 0.0])})")
    rows = store.ro_query(gkey, "CALL db.idx.vector.queryNodes('V', 'emb', 1, vecf32($q)) YIELD node RETURN node.gid", {"q": [0.9, 0.1, 0.0]})
    assert rows == [["a"]]


def test_algorithms_used_by_plan(store, gkey):
    store.query(gkey, """
      CREATE (a:St {gid:'a'}), (b:St {gid:'b'}), (c:St {gid:'c'}), (d:St {gid:'d'}),
             (a)-[:ACT {cost:1.0, risk_cost:0}]->(b), (b)-[:ACT {cost:1.0, risk_cost:0}]->(d),
             (a)-[:ACT {cost:0.5, risk_cost:100}]->(c), (c)-[:ACT {cost:0.5, risk_cost:0}]->(d)""")
    rows = store.ro_query(gkey, """
      MATCH (s:St {gid:'a'}), (t:St {gid:'d'})
      CALL algo.SPpaths({sourceNode: s, targetNode: t, relTypes: ['ACT'], weightProp: 'cost',
                         costProp: 'risk_cost', maxCost: 1, maxLen: 10, pathCount: 1})
      YIELD path, pathWeight
      RETURN [n IN nodes(path) | n.gid], pathWeight""")
    assert rows[0][0] == ["a", "b", "d"]
    assert store.ro_query(gkey, "CALL algo.pageRank('St', 'ACT') YIELD node, score RETURN count(node)") == [[4]]
    assert store.ro_query(gkey, "CALL algo.betweenness({nodeLabels:['St'], relationshipTypes:['ACT']}) YIELD node, score RETURN count(node)") == [[4]]
    assert store.ro_query(gkey, "CALL algo.labelPropagation({nodeLabels:['St'], relationshipTypes:['ACT']}) YIELD node, communityId RETURN count(node)") == [[4]]
    bfs = store.ro_query(gkey, "MATCH (s:St {gid:'a'}) CALL algo.bfs(s, 3, 'ACT') YIELD nodes RETURN [n IN nodes | n.gid]")
    assert set(bfs[0][0]) >= {"b", "c", "d"}


def test_graph_copy(store, gkey):
    store.query(gkey, "CREATE (:X {v:1})")
    store.copy(gkey, gkey + "@br:t")
    assert store.exists(gkey + "@br:t")
    assert store.ro_query(gkey + "@br:t", "MATCH (x:X) RETURN x.v") == [[1]]
