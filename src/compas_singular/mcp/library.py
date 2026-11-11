"""The guidance, workflows and worked examples the MCP server serves, as Markdown and JSON on disk.

Nothing here fails loudly; missing files give defaults.
"""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

import json
import os
from typing import Any
from typing import Sequence

from compas_singular.mcp.describe import DEFAULT_THRESHOLDS

__all__ = [
    "LIBRARY_ENVVAR",
    "library_root",
    "thresholds",
    "list_resources",
    "read_resource",
    "list_prompts",
    "get_prompt",
    "save_example",
    "load_examples",
    "fingerprint",
    "recall",
]


#: Points the library somewhere else -- a working copy, or a project's own.
LIBRARY_ENVVAR = "COMPAS_SINGULAR_MCP_LIBRARY"

GUIDANCE_SCHEME = "guidance://"
EXAMPLE_SCHEME = "example://"
SESSION_SCHEME = "session://"


def library_root() -> str:
    """The library directory: the environment override, or the shipped one."""
    override = (os.environ.get(LIBRARY_ENVVAR) or "").strip()
    if override:
        return override
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "library")


def _folder(name: str) -> str:
    return os.path.join(library_root(), name)


def _read(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8") as stream:
            return stream.read()
    except (IOError, OSError, ValueError, UnicodeDecodeError):
        return None


def _names_in(folder: str, suffix: str) -> list[str]:
    try:
        return sorted(n[: -len(suffix)] for n in os.listdir(folder) if n.endswith(suffix))
    except OSError:
        return []


def _summarise(text: str | None, limit: int = 160) -> str:
    """The first real sentence of a Markdown file, for a resource listing."""
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith(">"):
            continue
        line = line.replace("**", "").replace("`", "")
        return line[:limit]
    return ""


def thresholds() -> dict[str, Any]:
    """Quality bands, from ``thresholds.json``, over the built-in defaults.

    A partial file is merged per metric, so overriding ``min_angle`` alone does
    not silently drop the bands for everything else.
    """
    merged = dict((name, dict(limits)) for name, limits in DEFAULT_THRESHOLDS.items())
    raw = _read(os.path.join(library_root(), "thresholds.json"))
    if not raw:
        return merged
    try:
        loaded = json.loads(raw)
    except ValueError:
        return merged
    if not isinstance(loaded, dict):
        return merged
    for name, limits in loaded.items():
        if isinstance(limits, dict):
            merged.setdefault(name, {}).update(limits)
    return merged


def _unsafe(name: str) -> bool:
    """Whether a resource or prompt name would escape the library folder.

    The name comes off the wire, so it is not trusted. Built from ``os.sep``
    rather than a literal separator so it is right on both platforms.
    """
    if not name or name.startswith("."):
        return True
    return any(sep and sep in name for sep in ("/", os.sep, os.altsep))


# ==============================================================================
# resources
# ==============================================================================


def list_resources() -> list[dict[str, Any]]:
    """Everything readable, as MCP resource descriptors.

    ``session://current`` is NOT listed here -- it is live state, so the server
    adds it. The library only knows about files.
    """
    out = []
    guidance = _folder("guidance")
    for name in _names_in(guidance, ".md"):
        text = _read(os.path.join(guidance, name + ".md")) or ""
        out.append({"uri": GUIDANCE_SCHEME + name, "name": name.replace("_", " "), "description": _summarise(text), "mimeType": "text/markdown"})
    for record in load_examples():
        out.append({"uri": EXAMPLE_SCHEME + record["name"], "name": "example: " + record["name"], "description": _example_line(record), "mimeType": "text/markdown"})
    return out


def read_resource(uri: str) -> str | None:
    """The body behind a ``guidance://`` or ``example://`` uri. ``None`` if absent."""
    if uri.startswith(GUIDANCE_SCHEME):
        name = uri[len(GUIDANCE_SCHEME) :]
        if _unsafe(name):
            return None  # never escape the library folder
        return _read(os.path.join(_folder("guidance"), name + ".md"))
    if uri.startswith(EXAMPLE_SCHEME):
        name = uri[len(EXAMPLE_SCHEME) :]
        for record in load_examples():
            if record["name"] == name:
                return render_example(record)
        return None
    return None


# ==============================================================================
# prompts
# ==============================================================================


def list_prompts() -> list[dict[str, Any]]:
    folder = _folder("prompts")
    out = []
    for name in _names_in(folder, ".md"):
        text = _read(os.path.join(folder, name + ".md")) or ""
        out.append({"name": name, "description": _summarise(text), "arguments": []})
    return out


def get_prompt(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """One workflow, as a single user message.

    Returns ``None`` when there is no such prompt, which the protocol turns into
    an invalid-params error naming it.
    """
    if _unsafe(name):
        return None
    text = _read(os.path.join(_folder("prompts"), name + ".md"))
    if text is None:
        return None
    return {"description": _summarise(text), "messages": [{"role": "user", "content": {"type": "text", "text": text}}]}


# ==============================================================================
# worked examples
# ==============================================================================


def fingerprint(metrics: dict[str, Any] | None, walls: int = 0, faces: int | None = None) -> dict[str, Any]:
    """The geometric shape of a problem, for matching one session against another."""
    metrics = metrics or {}
    faces = faces if faces is not None else metrics.get("faces") or 0
    aspect = metrics.get("aspect_max")
    dominant = "none"
    limits = thresholds()
    from compas_singular.mcp.describe import band

    worst_band, dominant = "good", "none"
    for metric in ("min_angle", "max_angle", "aspect_max"):
        if metric not in limits:
            continue
        rating = band(metrics.get(metric), limits[metric])
        order = {"good": 0, "usable": 1, "poor": 2, "unusable": 3, "unknown": 0}
        if order.get(rating, 0) > order.get(worst_band, 0):
            worst_band, dominant = rating, metric
    return {"loops": int(walls), "faces_band": _faces_band(faces), "dominant": dominant, "severity": worst_band, "aspect_max": aspect}


def _faces_band(faces: int | None) -> str:
    faces = faces or 0
    for edge, name in ((50, "tiny"), (300, "small"), (2000, "medium")):
        if faces < edge:
            return name
    return "large"


def _score(a: dict[str, Any], b: dict[str, Any]) -> int:
    """How alike two fingerprints are. Higher is closer."""
    score = 0
    if a.get("loops") == b.get("loops"):
        score += 3
    if a.get("faces_band") == b.get("faces_band"):
        score += 2
    if a.get("dominant") == b.get("dominant"):
        score += 4
    if a.get("severity") == b.get("severity"):
        score += 1
    return score


def load_examples() -> list[dict[str, Any]]:
    """Every stored session. Malformed files are skipped, not raised on."""
    folder = _folder("sessions")
    out = []
    for name in _names_in(folder, ".json"):
        raw = _read(os.path.join(folder, name + ".json"))
        if not raw:
            continue
        try:
            record = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(record, dict):
            continue
        record.setdefault("name", name)
        out.append(record)
    return out


def _example_line(record: dict[str, Any]) -> str:
    """One line describing a stored example, for a listing."""
    before = record.get("before") or {}
    after = record.get("after") or {}
    return "{} -- {} steps, min angle {} -> {}, verdict {}".format(
        record.get("instruction") or record.get("title") or "no instruction recorded",
        len(record.get("steps") or []),
        _round(before.get("min_angle")),
        _round(after.get("min_angle")),
        record.get("verdict", "?"),
    )


def _round(value: Any) -> str:
    try:
        return "{:.1f}".format(float(value))
    except (TypeError, ValueError):
        return "?"


def render_example(record: dict[str, Any]) -> str:
    """A stored session as Markdown, which is what a model reads best."""
    lines = ["# {}".format(record.get("title") or record.get("name") or "example"), ""]
    verdict = record.get("verdict", "?")
    lines.append("**Verdict: {}**".format(verdict))
    if record.get("instruction"):
        lines.append("")
        lines.append("Asked: {}".format(record["instruction"]))
    before, after = record.get("before") or {}, record.get("after") or {}
    lines += [
        "",
        "| | min angle | max angle | aspect |",
        "|---|---|---|---|",
        "| before | {} | {} | {} |".format(_round(before.get("min_angle")), _round(before.get("max_angle")), _round(before.get("aspect_max"))),
        "| after | {} | {} | {} |".format(_round(after.get("min_angle")), _round(after.get("max_angle")), _round(after.get("aspect_max"))),
    ]
    steps = record.get("steps") or []
    if steps:
        lines += ["", "## What was done", ""]
        for i, step in enumerate(steps):
            detail = step.get("arguments") or {}
            shown = ", ".join("{}={}".format(k, v) for k, v in sorted(detail.items()))
            lines.append("{}. `{}({})`{}".format(i + 1, step.get("action", "?"), shown, " -- " + step["note"] if step.get("note") else ""))
    remarks = record.get("remarks") or []
    if remarks:
        lines += ["", "## Remarks", ""]
        lines += ["- {}".format(r if isinstance(r, str) else r.get("text", "")) for r in remarks]
    if record.get("lesson"):
        lines += ["", "## Lesson", "", record["lesson"]]
    return "\n".join(lines)


def save_example(record: dict[str, Any], directory: str | None = None) -> str:
    """Write a finished session into the corpus without overwriting. Returns the path."""
    folder = directory or _folder("sessions")
    if not os.path.isdir(folder):
        os.makedirs(folder)
    stem = _slug(record.get("name") or "session")
    name, index = stem, 2
    while os.path.exists(os.path.join(folder, name + ".json")):
        name = "{}-{}".format(stem, index)
        index += 1
    record = dict(record)
    record["name"] = name
    path = os.path.join(folder, name + ".json")
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, default=str)
    return path


def _slug(text: Any) -> str:
    out = []
    for character in str(text).strip().lower():
        if character.isalnum():
            out.append(character)
        elif character in " -_" and out and out[-1] != "-":
            out.append("-")
    return ("".join(out).strip("-") or "session")[:60]


def recall(target: dict[str, Any], limit: int = 3, verdicts: Sequence[str] = ("good", "acceptable", "bad")) -> list[dict[str, Any]]:
    """The stored examples closest to ``target``, best match first.

    Parameters
    ----------
    target : dict
        A fingerprint, from ``fingerprint``.
    limit : int, optional
    verdicts : sequence, optional
        Which verdicts to include. A bad example is instructive -- it is what
        NOT to repeat -- so it is included by default.

    Returns
    -------
    list[dict]
        ``{name, score, verdict, summary, uri}``, so the caller can read the
        full text through ``resources/read`` if a summary is not enough.
    """
    out = []
    for record in load_examples():
        if record.get("verdict") not in verdicts:
            continue
        stored = record.get("fingerprint") or {}
        out.append(
            {
                "name": record["name"],
                "score": _score(target or {}, stored),
                "verdict": record.get("verdict", "?"),
                "fingerprint": stored,
                "summary": _example_line(record),
                "uri": EXAMPLE_SCHEME + record["name"],
            }
        )
    out.sort(key=lambda item: (-item["score"], item["name"]))
    return out[: max(1, int(limit))]
