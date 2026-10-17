import json
import os
import time
import pytest
from sitegraph.blobs import BlobStore
from sitegraph.embed import CachedEmbedder, HashEmbedder
from sitegraph.llm import (ChatMessage, FakeLLM, OfflineLLM, ScriptedLLM, ToolCall, TracingLLM, can_call_tools,
                           has_model, unwrap_llm)
from sitegraph.maintenance import blob_gc


class Counting(HashEmbedder):
    def __init__(self):
        super().__init__(384)
        self.n = 0

    def embed(self, texts):
        self.n += len(texts)
        return super().embed(texts)


def test_cached_embedder_hits_disk(tmp_path):
    inner = Counting()
    e = CachedEmbedder(inner, tmp_path / "emb.sqlite")
    a = e.embed(["hello world", "hello world", "other"])
    assert inner.n == 2
    e2 = CachedEmbedder(inner, tmp_path / "emb.sqlite")
    assert e2.embed(["hello world"]) == [a[0]] and inner.n == 2
    assert e2.model_name == "Counting"           # the Site node records the model by this name, not "CachedEmbedder"


def test_tracing_llm_writes_jsonl(tmp_path):
    t = TracingLLM(FakeLLM(lambda s, u: "ok"), tmp_path)
    t.start_trace("run:abc")
    t.complete("sys", "user text")
    path = t.stop_trace()
    line = json.loads(path.read_text().splitlines()[0])
    assert line["user"] == "user text" and line["output"] == "ok" and t.calls == 1


def test_tracing_chat_redacts_sensitive_tool_arguments_and_never_stores_images(tmp_path):
    reply = ChatMessage("assistant", tool_calls=[ToolCall("1", "fill", {"element_id": "e1", "value": "hunter2-secret"})])
    inner = ScriptedLLM([reply, ChatMessage("assistant", content="done")])
    redact = lambda name, args: {k: ("<hidden>" if (name == "fill" and k == "value") else v) for k, v in args.items()}
    t = TracingLLM(inner, tmp_path, redact=redact)
    t.start_trace("run:chat")
    out = t.chat([ChatMessage("user", "type it")], [], image_png=b"\x89PNG-bytes")
    t.chat([ChatMessage("user", "type it"), out, ChatMessage("tool", '{"ok": true}', tool_call_id="1", name="fill")], [])
    text = t.stop_trace().read_text()
    assert "hunter2-secret" not in text and "<hidden>" in text and "PNG-bytes" not in text
    first = json.loads(text.splitlines()[0])
    assert first["kind"] == "chat" and first["image"] is True
    assert inner.calls[0][0][0].content == "type it" and inner.images[0] == b"\x89PNG-bytes"   # the model still got it
    # the second request already contains the first answer: its arguments are redacted there too
    assert "hunter2-secret" not in json.dumps(json.loads(text.splitlines()[1])["messages"])


def test_wrapping_never_disables_the_agents(tmp_path):
    scripted = ScriptedLLM([])
    assert can_call_tools(TracingLLM(scripted, tmp_path))
    assert not can_call_tools(TracingLLM(FakeLLM(), tmp_path))          # the inner model cannot call tools
    offline = TracingLLM(OfflineLLM(), tmp_path)
    assert unwrap_llm(offline).__class__ is OfflineLLM and not has_model(offline) and not can_call_tools(offline)
    assert not has_model(None)


@pytest.mark.integration
def test_blob_gc_keeps_referenced(store, gkey, tmp_path):
    blobs = BlobStore(tmp_path)
    keep = blobs.put_bytes(b"keep", ".txt")
    drop = blobs.put_bytes(b"drop", ".txt")
    store.query(gkey, "CREATE (:State {gid:'st:1', screenshot_ref:$r})", {"r": keep})
    old = time.time() - 7200
    os.utime(blobs.path(drop), (old, old))
    os.utime(blobs.path(keep), (old, old))
    assert blob_gc(store, blobs, [gkey]) == 1
    assert blobs.path(keep).exists() and not blobs.path(drop).exists()


@pytest.mark.integration
def test_a_run_stores_its_llm_trace_as_a_blob(store, gkey, settings, tmp_path):
    from sitegraph.agents.navigator import Navigator
    from sitegraph.schema import ensure_device_schema

    ensure_device_schema(store, gkey, 384)
    llm = TracingLLM(FakeLLM(lambda s, u: "{}"), tmp_path / "traces")
    blobs = BlobStore(tmp_path / "blobs")
    nav = Navigator(store, gkey, "x.test", None, settings, HashEmbedder(), llm=llm, blobs=blobs)
    run = nav._start("execute", "open cart")
    llm.complete("system", "what is on the page?")
    nav._finish(run, "done")
    ref = store.ro_query(gkey, "MATCH (r:Run {gid:$g}) RETURN r.trace_ref", {"g": run})[0][0]
    assert ref and json.loads(blobs.read(ref).decode().splitlines()[0])["user"] == "what is on the page?"
    quiet = Navigator(store, gkey, "x.test", None, settings, HashEmbedder(), llm=FakeLLM(), blobs=blobs)
    r2 = quiet._start("execute", "no tracing")
    quiet._finish(r2, "done")                                  # a plain model has no trace: nothing is stored
    assert store.ro_query(gkey, "MATCH (r:Run {gid:$g}) RETURN r.trace_ref", {"g": r2}) == [[None]]
