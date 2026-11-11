"""The request/response file protocol between the MCP server and Rhino: one directory, no locks.

Writes are temp-then-rename, claims are renames, and nothing blocks past its deadline.
"""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

import json
import os
import sys
import tempfile
import time
from typing import Any

__all__ = [
    "SPOOL_ENVVAR",
    "default_directory",
    "spool_directory",
    "new_id",
    "post",
    "wait",
    "pending",
    "take",
    "answer",
    "heartbeat",
    "clear_link",
    "link_state",
    "prune",
]


#: Overrides the spool location. Unlike the solve cache's variable, an empty
#: value is NOT meaningful -- there is no "disabled" spool, only a different one.
SPOOL_ENVVAR = "COMPAS_SINGULAR_MCP_SPOOL"

REQUEST_SUFFIX = ".req.json"
TAKEN_SUFFIX = ".req.taken"
RESPONSE_SUFFIX = ".resp.json"
LINK_NAME = "link.json"

#: How long a tool call waits for Rhino before giving up and saying so.
DEFAULT_TIMEOUT = 30.0

#: Poll interval while waiting. Small enough to feel immediate, large enough
#: that a 30 second wait is 600 stat calls rather than 30000.
POLL_SECONDS = 0.05

#: Requests, responses and claims older than this are litter from a crash.
DEFAULT_MAX_AGE = 900.0

#: How long past its caller's own timeout a request stays servable. Covers
#: clock jitter between posting and waiting; any real delay beyond it means the
#: caller has already been told the request failed.
EXPIRY_GRACE = 2.0

#: The link rewrites its heartbeat no more often than this.
HEARTBEAT_SECONDS = 1.0


def default_directory() -> str:
    """The per-user spool directory: ``$COMPAS_SINGULAR_MCP_SPOOL``, else a per-user cache or temp dir.

    Returns
    -------
    str
    """
    override = os.environ.get(SPOOL_ENVVAR)
    if override is not None:
        override = override.strip()
        if override:
            return override

    root = None
    if sys.platform == "win32":
        root = os.environ.get("LOCALAPPDATA")
    if not root:
        root = os.environ.get("XDG_RUNTIME_DIR")
    if not root:
        root = os.environ.get("XDG_CACHE_HOME")
    if not root:
        home = os.path.expanduser("~")
        if home and home != "~":
            root = os.path.join(home, ".cache")
    if not root:
        root = tempfile.gettempdir()
    return os.path.join(root, "compas_singular", "mcp-spool")


def spool_directory(directory: str | None = None) -> str:
    """The spool directory, created if it is not there yet.

    Parameters
    ----------
    directory : str, optional
        Use this instead of ``default_directory``. The tests pass a tmpdir.

    Returns
    -------
    str
    """
    path = directory or default_directory()
    if not os.path.isdir(path):
        try:
            os.makedirs(path)
        except OSError:
            # Lost a race with another process, or the parent is unwritable.
            # The isdir check tells the two apart.
            if not os.path.isdir(path):
                raise
    return path


# ==============================================================================
# reading and writing, atomically
# ==============================================================================


def _write_json(path: str, payload: Any) -> None:
    """Write ``payload`` to ``path`` so no reader can ever see it half-written."""
    folder = os.path.dirname(path)
    handle, temporary = tempfile.mkstemp(dir=folder, suffix=".tmp")
    try:
        with os.fdopen(handle, "w") as stream:
            json.dump(payload, stream, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.remove(temporary)
        except OSError:
            pass
        raise


def _read_json(path: str) -> Any:
    """``path`` decoded, or ``None`` if it is gone or not yet readable."""
    try:
        with open(path, "r") as stream:
            return json.load(stream)
    except (IOError, OSError, ValueError):
        # ValueError covers a file that exists but is not valid JSON, which on a
        # filesystem without atomic rename would be a partial write. Treating it
        # as "not ready" rather than raising keeps the poll loop simple.
        return None


_COUNTER = [0]


def new_id() -> str:
    """A request id that sorts lexically into arrival order (timestamp plus counter).

    Returns
    -------
    str
    """
    _COUNTER[0] += 1
    now = time.time()
    stamp = time.strftime("%Y%m%d%H%M%S", time.localtime(now))
    micro = int((now % 1.0) * 1000000)
    return "{}{:06d}-{:04d}".format(stamp, micro, _COUNTER[0] % 10000)


# ==============================================================================
# the server side
# ==============================================================================


def post(verb: str, args: dict[str, Any] | None = None, directory: str | None = None, ttl: float | None = None) -> str:
    """Put a request on the spool and return its id.

    Parameters
    ----------
    verb : {'ping', 'pull', 'push'}
        What Rhino should do. The link understands nothing else.
    args : dict, optional
    directory : str, optional
    ttl : float, optional
        Seconds the request stays servable -- pass the timeout it will be waited
        for with. ``take`` refuses it once ``ttl`` plus
        ``EXPIRY_GRACE`` has passed. ``None`` never expires.

    Returns
    -------
    str
        The request id, for ``wait``.
    """
    folder = spool_directory(directory)
    request_id = new_id()
    now = time.time()
    payload = {"id": request_id, "verb": verb, "args": dict(args or {}), "created": now}
    if ttl is not None:
        payload["expires"] = now + max(0.0, float(ttl)) + EXPIRY_GRACE
    _write_json(os.path.join(folder, request_id + REQUEST_SUFFIX), payload)
    return request_id


def wait(request_id: str, timeout: float = DEFAULT_TIMEOUT, directory: str | None = None) -> dict[str, Any]:
    """Poll for a response until it arrives or ``timeout`` passes; never raises.

    A timeout withdraws the request and returns a refusal saying why.

    Parameters
    ----------
    request_id : str
    timeout : float, optional
    directory : str, optional

    Returns
    -------
    dict
        ``{'ok': True, 'result': ...}`` on success, or ``{'ok': False,
        'reason': ...}`` -- from Rhino if it refused, or synthesised here if the
        wait ran out. A synthesised refusal carries ``timed_out``, ``link`` and
        ``queued`` so the caller can say WHY nothing came back, and
        ``in_progress`` -- true when Rhino had already claimed the request, so
        it could not be withdrawn and may yet complete.
    """
    folder = spool_directory(directory)
    path = os.path.join(folder, request_id + RESPONSE_SUFFIX)
    deadline = time.time() + max(0.0, timeout)

    while True:
        payload = _consume(path)
        if payload is not None:
            return payload
        if time.time() >= deadline:
            break
        time.sleep(POLL_SECONDS)

    # Withdraw the request before reporting, so Rhino cannot act on it after the
    # caller has been told it failed. The removal is the claim: if ``take`` got
    # there first this raises, and the request is Rhino's to finish.
    try:
        os.remove(os.path.join(folder, request_id + REQUEST_SUFFIX))
    except OSError:
        pass
    # It may have been answered between the last poll and the withdrawal.
    payload = _consume(path)
    if payload is not None:
        return payload
    in_progress = os.path.exists(os.path.join(folder, request_id + TAKEN_SUFFIX))

    # Nothing came back. Say which of the two reasons it was, because the fix is
    # different: attach the link, or wait for Rhino to leave the command it is in.
    state = link_state(folder)
    queued = len(pending(folder))
    if state is None:
        reason = "Rhino is not listening: no link is attached. Run CMD_mcp_link in Rhino, then try again."
    else:
        reason = "Rhino is attached but did not answer within {:g}s. It is most likely inside a command -- the link waits for that to finish before touching the document.".format(
            timeout
        )
    if in_progress:
        reason += " Rhino had ALREADY TAKEN this request, so it could not be withdrawn and may still complete -- check the document before retrying anything that writes to it."
    else:
        reason += " The request was withdrawn; Rhino will not act on it."
    return {"ok": False, "reason": reason, "timed_out": True, "in_progress": in_progress, "link": state, "queued": queued}


def _consume(path: str) -> Any:
    """Read a response and delete it, so it is never replayed. ``None`` if absent."""
    payload = _read_json(path)
    if payload is not None:
        try:
            os.remove(path)
        except OSError:
            pass
    return payload


def pending(directory: str | None = None) -> list[str]:
    """The ids of requests posted and not yet taken, oldest first.

    Returns
    -------
    list[str]
    """
    folder = spool_directory(directory)
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    return sorted(name[: -len(REQUEST_SUFFIX)] for name in names if name.endswith(REQUEST_SUFFIX))


# ==============================================================================
# the Rhino side
# ==============================================================================


def take(directory: str | None = None) -> tuple[str, str, dict[str, Any]] | None:
    """Claim the oldest unexpired request by renaming it. ``(request_id, verb, args)`` or ``None``.

    Returns
    -------
    tuple or None
        ``(request_id, verb, args)``.
    """
    folder = spool_directory(directory)
    now = time.time()
    for request_id in pending(folder):
        source = os.path.join(folder, request_id + REQUEST_SUFFIX)
        claimed = os.path.join(folder, request_id + TAKEN_SUFFIX)
        try:
            os.replace(source, claimed)
        except OSError:
            continue
        payload = _read_json(claimed)
        if payload is None:
            # Claimed but unreadable -- drop it rather than leave a job that can
            # never be answered and never be retried.
            try:
                os.remove(claimed)
            except OSError:
                pass
            continue
        expires = payload.get("expires")
        if expires is not None and now > expires:
            # Nobody is waiting, so an answer would only be litter.
            try:
                os.remove(claimed)
            except OSError:
                pass
            continue
        return (request_id, payload.get("verb"), payload.get("args") or {})
    return None


def answer(request_id: str, ok: bool, result: Any = None, error: str | None = None, directory: str | None = None) -> None:
    """Write the response, then release the claim.

    Parameters
    ----------
    request_id : str
    ok : bool
    result : optional
        Anything JSON-serialisable. Ignored when ``ok`` is false.
    error : str, optional
        Why it failed. Ignored when ``ok`` is true.
    directory : str, optional
    """
    folder = spool_directory(directory)
    payload = {"id": request_id, "ok": bool(ok), "completed": time.time()}
    if ok:
        payload["result"] = result
    else:
        payload["reason"] = error or "the link refused, without saying why"
    _write_json(os.path.join(folder, request_id + RESPONSE_SUFFIX), payload)
    try:
        os.remove(os.path.join(folder, request_id + TAKEN_SUFFIX))
    except OSError:
        pass


_LAST_BEAT = [0.0]


def heartbeat(document: str | None = None, directory: str | None = None, force: bool = False) -> None:
    """Record that the link is alive, at most once a second unless ``force``.

    Parameters
    ----------
    document : str, optional
        The open document's name, so the server can say WHICH Rhino.
    directory : str, optional
    force : bool, optional
        Write even if the last beat was recent. Used when attaching.
    """
    now = time.time()
    if not force and (now - _LAST_BEAT[0]) < HEARTBEAT_SECONDS:
        return
    _LAST_BEAT[0] = now
    folder = spool_directory(directory)
    path = os.path.join(folder, LINK_NAME)
    started = now
    existing = _read_json(path)
    if existing and existing.get("pid") == os.getpid():
        started = existing.get("started", now)
    _write_json(path, {"pid": os.getpid(), "document": document, "started": started, "last_seen": now})


def clear_link(directory: str | None = None) -> None:
    """Remove the heartbeat, so the server reports the link as detached."""
    folder = spool_directory(directory)
    try:
        os.remove(os.path.join(folder, LINK_NAME))
    except OSError:
        pass
    _LAST_BEAT[0] = 0.0


def link_state(directory: str | None = None) -> dict[str, Any] | None:
    """Who is attached, and how long ago they said so.

    Returns
    -------
    dict or None
        ``None`` when no link has ever attached. Otherwise the heartbeat plus
        ``age`` in seconds and ``stale``, which is true once the link has been
        quiet for more than a few beats -- Rhino closed without detaching, or is
        wedged inside a long command.
    """
    folder = spool_directory(directory)
    state = _read_json(os.path.join(folder, LINK_NAME))
    if not state:
        return None
    age = max(0.0, time.time() - state.get("last_seen", 0.0))
    state["age"] = age
    state["stale"] = age > (HEARTBEAT_SECONDS * 10)
    return state


# ==============================================================================
# housekeeping
# ==============================================================================


def prune(max_age: float = DEFAULT_MAX_AGE, directory: str | None = None) -> int:
    """Delete requests, claims and responses older than ``max_age`` seconds.

    Litter from a crash on either side. The heartbeat is never pruned -- it is
    rewritten in place and its staleness is reported rather than acted on.

    Returns
    -------
    int
        How many files were removed.
    """
    folder = spool_directory(directory)
    try:
        names = os.listdir(folder)
    except OSError:
        return 0
    cutoff = time.time() - max_age
    removed = 0
    for name in names:
        if not (name.endswith(REQUEST_SUFFIX) or name.endswith(TAKEN_SUFFIX) or name.endswith(RESPONSE_SUFFIX)):
            continue
        path = os.path.join(folder, name)
        try:
            if os.path.getmtime(path) >= cutoff:
                continue
            os.remove(path)
            removed += 1
        except OSError:
            continue
    return removed
