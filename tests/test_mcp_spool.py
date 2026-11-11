"""The MCP spool: round trip, atomic claim, ordering, timeouts, crash litter."""

import json
import os

import pytest

from compas_singular.mcp.bridge import spool


@pytest.fixture(autouse=True)
def _isolated(mcp_spool):
    return mcp_spool


def test_a_request_goes_out_and_an_answer_comes_back():
    assert spool.link_state() is None
    rid = spool.post("pull", {"layer": "QuadMesh"})
    assert spool.pending() == [rid]
    assert spool.take() == (rid, "pull", {"layer": "QuadMesh"})
    assert spool.pending() == []
    assert spool.take() is None
    spool.answer(rid, True, result={"faces": 12})
    got = spool.wait(rid, timeout=2)
    assert got.get("ok") and got["result"] == {"faces": 12}


def test_ids_sort_into_arrival_order_and_take_is_oldest_first():
    ids = [spool.post("ping") for _ in range(5)]
    assert spool.pending() == ids
    assert spool.take()[0] == ids[0]
    for _ in range(4):
        spool.take()
    assert spool.pending() == []


def test_a_timeout_is_a_refusal_that_names_the_cause():
    missing = spool.wait("nosuchid", timeout=0.2)
    assert missing.get("timed_out") is True
    assert "not listening" in missing["reason"]
    spool.heartbeat("plate.3dm", force=True)
    state = spool.link_state()
    assert state["document"] == "plate.3dm"
    assert state["stale"] is False
    busy = spool.wait("nosuchid", timeout=0.2)
    assert "inside a command" in busy["reason"]


def test_a_refusal_from_rhino_survives_the_wire_and_is_consumed_once():
    rid = spool.post("pull")
    spool.take()
    spool.answer(rid, False, error="No mesh on that layer.")
    bad = spool.wait(rid, timeout=2)
    assert bad.get("ok") is False
    assert bad["reason"] == "No mesh on that layer."
    assert spool.wait(rid, timeout=0.2).get("timed_out") is True


def test_crash_litter_is_pruned_but_the_heartbeat_is_not():
    spool.heartbeat("plate.3dm", force=True)
    spool.post("ping")
    spool.post("ping")
    spool.take()  # leave a .req.taken behind
    assert spool.prune(max_age=-1) >= 2
    assert spool.link_state() is not None
    spool.clear_link()
    assert spool.link_state() is None


def test_writes_are_atomic(mcp_spool):
    rid = spool.post("push", {"mesh": {"vertices": [[0, 0, 0]] * 500}})
    with open(os.path.join(mcp_spool, rid + ".req.json")) as stream:
        assert isinstance(json.load(stream), dict)
    assert [n for n in os.listdir(mcp_spool) if n.endswith(".tmp")] == []


def test_a_timed_out_request_is_withdrawn_so_it_cannot_fire_late():
    rid = spool.post("push", {"layer": "QuadMesh"}, ttl=0.2)
    out = spool.wait(rid, timeout=0.2)
    assert out.get("timed_out") is True
    assert rid not in spool.pending()
    served = []
    while True:
        job = spool.take()
        if job is None:
            break
        served.append(job[0])
    assert rid not in served
    assert "withdrawn" in out["reason"] and out["in_progress"] is False


def test_a_request_rhino_already_took_cannot_be_withdrawn_and_says_so():
    rid = spool.post("push", ttl=0.2)
    spool.take()  # Rhino has it, and is slow
    out = spool.wait(rid, timeout=0.2)
    assert out.get("in_progress") is True
    assert "ALREADY TAKEN" in out["reason"]


def test_an_expired_request_is_dropped_by_take_never_served(mcp_spool):
    rid = spool.post("push", ttl=0.0)
    path = os.path.join(mcp_spool, rid + ".req.json")
    with open(path) as stream:
        stale = json.load(stream)
    stale["expires"] = stale["created"] - 1.0  # a server that died mid-wait
    spool._write_json(path, stale)
    fresh = spool.post("ping", ttl=30)
    job = spool.take()
    assert job and job[0] == fresh
    assert not os.path.exists(os.path.join(mcp_spool, rid + ".req.taken"))
    assert spool.post("ping") and spool.take() is not None
