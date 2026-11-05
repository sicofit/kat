"""Shared test fixtures and lightweight doubles for the KAT test suite.

The suite is designed to run without installing the package: this module
prepends the repository ``src`` directory to ``sys.path`` so ``import kat``
resolves against the working tree. It also provides a scripted stand-in for the
LLM client and a small tool registry whose binary is the current interpreter,
which lets the real executor run on any platform without external tooling.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

# Make the in-tree package importable whether or not it has been installed.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pytest

from kat.tools.base import Registry, ToolSpec


class MockClient:
    """A scripted replacement for :class:`kat.client.LLMClient`.

    It is constructed with the assistant turns the agent should receive, in
    order. Each call to :meth:`chat` consumes the next turn and returns an
    object exposing it as ``.text``, mirroring ``ChatResult``. Running out of
    scripted turns is a test authoring error and raises ``AssertionError``.
    """

    def __init__(self, turns):
        self.turns = list(turns)
        self.calls: list[list[dict]] = []

    def chat(self, messages, *, stream=None, on_token=None):
        assert self.turns, "MockClient ran out of scripted turns"
        self.calls.append(list(messages))
        text = self.turns.pop(0)
        if on_token is not None:
            on_token(text)
        return SimpleNamespace(text=text)


# A tool whose binary is the running interpreter. ``sys.executable`` is an
# absolute path, so ``shutil.which`` resolves it and ``available()`` is True on
# every platform. The builder turns ``code`` into ``python -c <code>``, giving
# deterministic output without depending on any installed security tool.
PYRUN = ToolSpec(
    name="pyrun",
    binary=sys.executable,
    summary="run a Python snippet",
    category="test",
    active=False,
    builder=lambda a: ["-c", a.get("code", "")],
)

# The same mechanism, but marked active so approval-policy behaviour can be
# exercised. Both tools share the one interpreter binary.
PYRUN_ACTIVE = ToolSpec(
    name="pyrun_active",
    binary=sys.executable,
    summary="run a Python snippet that requires approval",
    category="test",
    active=True,
    builder=lambda a: ["-c", a.get("code", "")],
)


def pyrun_registry() -> Registry:
    """A registry holding the interpreter-backed tools used across the suite."""
    return Registry([PYRUN, PYRUN_ACTIVE])


@pytest.fixture
def make_client():
    """Return the ``MockClient`` class for use as a factory: ``make_client(turns)``."""
    return MockClient


@pytest.fixture
def registry():
    """A fresh registry with the ``pyrun`` and ``pyrun_active`` tools."""
    return pyrun_registry()


@pytest.fixture
def pyrun_spec():
    """The inactive interpreter-backed tool spec."""
    return PYRUN


@pytest.fixture
def pyrun_active_spec():
    """The active interpreter-backed tool spec (subject to approval)."""
    return PYRUN_ACTIVE
