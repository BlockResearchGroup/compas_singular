"""The MCP library: guidance, prompts, thresholds and the example corpus."""

import json
import os

import pytest

from compas_singular.mcp import library

RECORD = {
    "name": "disc",
    "title": "A disc",
    "verdict": "good",
    "instruction": "improve it",
    "lesson": "relax was enough",
    "before": {"min_angle": 12.0, "max_angle": 170.0, "aspect_max": 9.0},
    "after": {"min_angle": 44.0, "max_angle": 130.0, "aspect_max": 2.0},
    "fingerprint": {"loops": 2, "faces_band": "small", "dominant": "min_angle", "severity": "unusable"},
    "steps": [{"action": "relax", "arguments": {"seams": "free"}}],
    "remarks": ["the hole drove everything"],
}


def test_the_shipped_library_loads(monkeypatch):
    monkeypatch.delenv(library.LIBRARY_ENVVAR, raising=False)
    assert os.path.isdir(library.library_root())
    names = [r["uri"] for r in library.list_resources()]
    for wanted in ("guidance://quality", "guidance://smoothing", "guidance://remarks"):
        assert wanted in names
    prompts = [p["name"] for p in library.list_prompts()]
    assert "improve_pulled_mesh" in prompts and "diagnose_only" in prompts
    assert all(r["description"] for r in library.list_resources())
    body = library.read_resource("guidance://smoothing")
    assert body and len(body) > 500
    assert "accepted: null" in body and "pin_singularities" in body


def test_thresholds_are_data(monkeypatch):
    monkeypatch.delenv(library.LIBRARY_ENVVAR, raising=False)
    limits = library.thresholds()
    assert limits["min_angle"]["good"] == 45.0
    assert set(limits) >= {"min_angle", "max_angle", "aspect_max", "share_below"}


def test_an_empty_override_lists_nothing_and_falls_back_on_thresholds(mcp_library):
    assert library.library_root() == mcp_library
    assert library.list_resources() == [] and library.list_prompts() == []
    assert library.thresholds()["min_angle"]["good"] == 45.0


def test_thresholds_merge_per_metric(mcp_library):
    with open(os.path.join(mcp_library, "thresholds.json"), "w") as stream:
        json.dump({"min_angle": {"good": 60.0}}, stream)
    merged = library.thresholds()
    assert merged["min_angle"]["good"] == 60.0
    assert merged["min_angle"]["usable"] == 25.0  # the rest of that metric's bands
    assert merged["aspect_max"]["good"] == 2.0  # and the other metrics


def test_a_malformed_thresholds_file_falls_back(mcp_library):
    with open(os.path.join(mcp_library, "thresholds.json"), "w") as stream:
        stream.write("{ this is not json")
    assert library.thresholds()["min_angle"]["good"] == 45.0


@pytest.mark.parametrize("uri", ["guidance://../../../secrets", "guidance://.hidden", "guidance://sub/dir", "evil://x"])
def test_a_resource_name_cannot_escape_the_folder(uri):
    assert library.read_resource(uri) is None


@pytest.mark.parametrize("name", ["../../etc/passwd", ".hidden", "sub/dir"])
def test_a_prompt_name_cannot_escape_the_folder(name):
    assert library.get_prompt(name) is None


def test_examples_save_render_and_list(mcp_library):
    assert os.path.isfile(library.save_example(RECORD))
    assert "example://disc" in [r["uri"] for r in library.list_resources()]
    text = library.read_resource("example://disc")
    assert text.startswith("# A disc")
    assert "**Verdict: good**" in text
    assert "| before |" in text and "| after |" in text
    assert "`relax(seams=free)`" in text
    assert "the hole drove everything" in text
    assert "relax was enough" in text


def test_recall_ranks_by_the_shape_of_the_problem(mcp_library):
    library.save_example(RECORD)
    library.save_example(dict(RECORD, name="near", fingerprint=dict(RECORD["fingerprint"])))
    library.save_example(dict(RECORD, name="far", fingerprint={"loops": 9, "faces_band": "large", "dominant": "aspect_max", "severity": "good"}))
    ranked = library.recall(RECORD["fingerprint"], limit=5)
    assert ranked[0]["fingerprint"] == RECORD["fingerprint"]
    assert ranked[0]["score"] > ranked[-1]["score"]
    assert library.recall(RECORD["fingerprint"], verdicts=("bad",)) == []
    assert os.path.basename(library.save_example(RECORD)) == "disc-2.json"


def test_a_malformed_example_is_skipped_not_raised_on(mcp_library):
    library.save_example(RECORD)
    library.save_example(dict(RECORD, name="near"))
    with open(os.path.join(mcp_library, "sessions", "broken.json"), "w") as stream:
        stream.write("{ nope")
    names = [r["name"] for r in library.load_examples()]
    assert "broken" not in names
    assert len(names) >= 2
