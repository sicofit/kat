"""Offline evaluation harness for the KAT agent loop.

This runs a handful of scripted scenarios end to end against the real agent,
registry, and executor, with the language model replaced by a deterministic
stand-in. It has no third-party dependencies and prints a plain PASS/FAIL line
per case followed by a summary.

Usage:
    python evals/run_eval.py

Each case in ``cases.json`` lists the assistant turns the model should emit in
order. A turn that is an action block triggers a tool call and is followed on
the next step by the turn after it; a turn with no action block ends the run as
the final answer. Expectations check the final answer text and whether the
``pyrun`` tool actually executed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

# Resolve the in-tree package without requiring an install.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from kat.agent import Agent
from kat.config import Config
from kat.session import Session
from kat.tools import Registry, ToolSpec


class MockClient:
    """Deterministic stand-in for the LLM client, driven by scripted turns."""

    def __init__(self, turns):
        self.turns = list(turns)

    def chat(self, messages, *, stream=None, on_token=None):
        assert self.turns, "scripted turns exhausted before the run finished"
        text = self.turns.pop(0)
        if on_token is not None:
            on_token(text)
        return SimpleNamespace(text=text)


def build_eval_registry() -> Registry:
    """A registry with a single interpreter-backed tool named ``pyrun``.

    Using the current interpreter keeps the harness self-contained: the tool
    resolves and runs on any platform and produces deterministic output.
    """
    pyrun = ToolSpec(
        name="pyrun",
        binary=sys.executable,
        summary="run a Python snippet",
        category="eval",
        active=False,
        builder=lambda a: ["-c", a.get("code", "")],
    )
    return Registry([pyrun])


def _tool_ran(session: Session) -> bool:
    """True when a real pyrun observation was fed back into the conversation."""
    return any(
        message["role"] == "user" and "OBSERVATION [pyrun]" in message["content"]
        for message in session.messages
    )


def _evaluate(case, registry) -> tuple[bool, str]:
    client = MockClient(case["turns"])
    config = Config(approval="none", stream=False)
    agent = Agent(client, registry, config, reporter=None)
    session = Session(system="evaluation harness")

    try:
        final = agent.run(session, case["user"])
    except Exception as exc:  # a crash is always a failure, never fatal here
        return False, f"run raised {exc!r}"

    expect = case.get("expect", {})
    reasons: list[str] = []

    needle = expect.get("final_contains")
    if needle is not None and needle not in final:
        reasons.append(f"final answer does not contain {needle!r}")

    want_ran = expect.get("tool_ran")
    if want_ran is not None:
        ran = _tool_ran(session)
        if ran != want_ran:
            reasons.append(f"tool_ran was {ran}, expected {want_ran}")

    if reasons:
        return False, "; ".join(reasons)
    return True, "ok"


def main() -> int:
    here = Path(__file__).resolve().parent
    cases = json.loads((here / "cases.json").read_text(encoding="utf-8"))
    registry = build_eval_registry()

    passed = 0
    for case in cases:
        ok, reason = _evaluate(case, registry)
        if ok:
            passed += 1
            print(f"PASS  {case['name']}")
        else:
            print(f"FAIL  {case['name']}: {reason}")

    total = len(cases)
    print(f"\n{passed}/{total} cases passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
