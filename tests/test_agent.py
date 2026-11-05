"""Tests for action parsing, result rendering, and the agent loop.

The loop tests use a scripted client and the interpreter-backed registry, so a
tool genuinely runs through the real executor. The emphasis is on the property
that matters most for this agent: an observation reflects the actual command
output, never text the model produced.
"""

from __future__ import annotations

from types import SimpleNamespace

from kat.agent import (
    MAX_ACTIONS_PER_TURN,
    Action,
    Agent,
    _is_context_overflow,
    parse_action,
    parse_actions,
    render_result,
)
from kat.client import LLMError
from kat.config import Config
from kat.session import Session
from kat.tools import ToolResult


# --- parse_action ----------------------------------------------------------


def test_parse_action_from_action_fence():
    text = '```action\n{"tool": "pyrun", "args": {"code": "print(1)"}, "reason": "demo"}\n```'
    action = parse_action(text)
    assert isinstance(action, Action)
    assert action.tool == "pyrun"
    assert action.args == {"code": "print(1)"}
    assert action.reason == "demo"


def test_parse_action_from_json_fence():
    text = '```json\n{"tool": "pyrun", "args": {"code": "print(1)"}}\n```'
    action = parse_action(text)
    assert isinstance(action, Action)
    assert action.tool == "pyrun"
    assert action.args == {"code": "print(1)"}
    assert action.reason == ""


def test_parse_action_from_bare_object():
    text = '{"tool": "pyrun", "args": {"code": "print(1)"}}'
    action = parse_action(text)
    assert isinstance(action, Action)
    assert action.tool == "pyrun"


def test_parse_action_prose_returns_none():
    text = "The service on the target responds on the standard web port."
    assert parse_action(text) is None


def test_parse_action_malformed_json_returns_none():
    fenced = '```action\n{"tool": "pyrun", "args": }\n```'
    bare = '{"tool": "pyrun" and then nonsense}'
    assert parse_action(fenced) is None
    assert parse_action(bare) is None


def test_parse_action_without_tool_key_returns_none():
    assert parse_action('```action\n{"args": {"code": "x"}}\n```') is None


def test_parse_action_with_non_dict_args_returns_none():
    assert parse_action('```action\n{"tool": "pyrun", "args": [1, 2]}\n```') is None


# --- render_result ---------------------------------------------------------


def test_render_result_success_includes_exit_and_stdout():
    result = ToolResult(tool="pyrun", argv=["python", "-c", "..."], returncode=0, stdout="hello\n")
    rendered = render_result(result)
    assert "exit code: 0" in rendered
    assert "stdout:" in rendered
    assert "hello" in rendered
    assert "timed out" not in rendered


def test_render_result_error():
    result = ToolResult(tool="x", argv=[], error="boom")
    assert render_result(result) == "error: boom"


def test_render_result_timed_out():
    result = ToolResult(tool="x", argv=["slow"], returncode=None, stdout="partial", timed_out=True)
    assert "(timed out)" in render_result(result)


def test_render_result_no_output():
    result = ToolResult(tool="x", argv=["quiet"], returncode=0, stdout="", stderr="")
    assert "(no output)" in render_result(result)


# --- the agent loop --------------------------------------------------------


def test_run_executes_tool_and_returns_final_answer(make_client, registry):
    turns = [
        '```action\n{"tool": "pyrun", "args": {"code": "print(1000+337)"}, "reason": "compute"}\n```',
        "The tool reported 1337. Analysis complete.",
    ]
    client = make_client(turns)
    agent = Agent(client, registry, Config(approval="none"))
    session = Session(system="profile")

    final = agent.run(session, "run the check and report the number")

    # The returned string is the model's final, action-free turn.
    assert final == "The tool reported 1337. Analysis complete."

    # Exactly one observation, carrying the real stdout, was fed back.
    observations = [
        message["content"]
        for message in session.messages
        if message["role"] == "user" and "OBSERVATION [pyrun]" in message["content"]
    ]
    assert len(observations) == 1
    assert "1337" in observations[0]

    # Two model turns were consumed: the action and the final answer.
    assert len(client.calls) == 2


def test_run_respects_declined_approval(make_client, registry):
    turns = [
        '```action\n{"tool": "pyrun_active", "args": {"code": "print(9*9)"}, "reason": "attempt"}\n```',
        "I will not proceed without explicit approval.",
    ]
    client = make_client(turns)
    agent = Agent(
        client,
        registry,
        Config(approval="active"),
        approver=lambda spec, action: False,
    )
    session = Session()

    final = agent.run(session, "use the active tool")

    assert final == "I will not proceed without explicit approval."

    transcript = "\n".join(message["content"] for message in session.messages)
    assert "OBSERVATION [pyrun_active]" in transcript
    assert "operator declined" in transcript
    # The tool must not have run: no execution result and no computed output.
    assert "exit code:" not in transcript
    assert "81" not in transcript


def test_observation_comes_from_executor_not_model(make_client, registry):
    # The action asks for 6*7; the real stdout is 42. The model text never
    # contains 42, so finding it in the observation proves the executor ran.
    turns = [
        '```action\n{"tool": "pyrun", "args": {"code": "print(6*7)"}, "reason": "compute"}\n```',
        "The computation has finished.",
    ]
    client = make_client(turns)
    agent = Agent(client, registry, Config(approval="none"))
    session = Session()

    agent.run(session, "compute the product")

    observations = [
        message["content"]
        for message in session.messages
        if message["role"] == "user" and "OBSERVATION [pyrun]" in message["content"]
    ]
    assert observations
    assert any("42" in observation for observation in observations)

    assistant_text = "\n".join(
        message["content"] for message in session.messages if message["role"] == "assistant"
    )
    assert "6*7" in assistant_text  # the model only ever referenced the expression
    assert "42" not in assistant_text  # the computed value came from execution alone


# --- the anti-loop guard ---------------------------------------------------
#
# A weak model tends to fall into a degenerate loop, re-issuing an identical
# call it already ran. The guard breaks that: a call whose (tool, args) already
# succeeded is not run a second time; instead the loop feeds back an honest note
# that the result is already in the transcript and a new step is needed. The
# note is true (no fabricated tool output), and a call that failed is still
# allowed to be retried.


def _observations(session, tool):
    return [
        message["content"]
        for message in session.messages
        if message["role"] == "user" and f"OBSERVATION [{tool}]" in message["content"]
    ]


def test_identical_successful_call_is_not_rerun(make_client, registry):
    repeated = '```action\n{"tool": "pyrun", "args": {"code": "print(7)"}, "reason": "again"}\n```'
    turns = [repeated, repeated, "The value is 7. Done."]
    client = make_client(turns)
    agent = Agent(client, registry, Config(approval="none"))
    session = Session()

    final = agent.run(session, "read the value")

    assert final == "The value is 7. Done."
    observations = _observations(session, "pyrun")
    # Both model turns produced an observation, but only the first one ran the
    # tool; the second is the guard's note, carrying no fresh execution output.
    assert len(observations) == 2
    assert sum("exit code:" in obs for obs in observations) == 1
    assert sum("already ran earlier" in obs for obs in observations) == 1
    assert len(client.calls) == 3


def test_malformed_action_json_is_nudged_not_ended(make_client, registry):
    # A weak model often emits a tool call with broken JSON (a stray or missing
    # brace). That must not end the run as if it were a final prose answer; the
    # loop feeds back a note asking for valid JSON and carries on.
    malformed = '```action\n{"tool": "pyrun", "args": }\n```'
    turns = [malformed, "Recovered and finished the analysis."]
    client = make_client(turns)
    agent = Agent(client, registry, Config(approval="none"))
    session = Session()

    final = agent.run(session, "do the thing")

    assert final == "Recovered and finished the analysis."
    nudges = [
        message["content"]
        for message in session.messages
        if message["role"] == "user" and "OBSERVATION [kat]" in message["content"]
    ]
    assert len(nudges) == 1
    assert "could not be parsed" in nudges[0]
    assert len(client.calls) == 2


def test_prose_answer_without_braces_still_ends_the_run(make_client, registry):
    turns = ["The target exposes FTP, SSH, and HTTP. No further action needed."]
    client = make_client(turns)
    agent = Agent(client, registry, Config(approval="none"))
    session = Session()

    final = agent.run(session, "summarise")

    assert final == "The target exposes FTP, SSH, and HTTP. No further action needed."
    assert len(client.calls) == 1


def test_a_failed_call_may_be_retried(make_client, registry):
    failing = (
        '```action\n{"tool": "pyrun", "args": {"code": "import sys; sys.exit(3)"}, '
        '"reason": "try"}\n```'
    )
    turns = [failing, failing, "Both attempts failed with exit 3."]
    client = make_client(turns)
    agent = Agent(client, registry, Config(approval="none"))
    session = Session()

    agent.run(session, "attempt the operation twice")

    observations = _observations(session, "pyrun")
    # A non-zero exit is not recorded as seen, so the identical retry runs for
    # real: two execution results, and the guard never fires.
    assert sum("exit code: 3" in obs for obs in observations) == 2
    assert all("already ran earlier" not in obs for obs in observations)


# --- several action blocks in one reply ------------------------------------
#
# The local model tends to batch independent probes into a single reply. Each
# block is run in order and yields its own real observation, so the model's
# plan is honoured instead of silently dropping all but the first. A reply with
# more than the per-turn cap runs the first few and is told the rest were held.


def test_parse_actions_returns_every_block_in_order():
    text = (
        '```action\n{"tool": "a", "args": {"x": 1}}\n```\n'
        "some prose in between\n"
        '```action\n{"tool": "b", "args": {}}\n```'
    )
    actions = parse_actions(text)
    assert [a.tool for a in actions] == ["a", "b"]
    assert actions[0].args == {"x": 1}


def test_parse_actions_is_empty_for_prose():
    assert parse_actions("No tools needed; the service is plain HTTP.") == []


def test_multiple_action_blocks_in_one_reply_all_run(make_client, registry):
    two = (
        '```action\n{"tool": "pyrun", "args": {"code": "print(11)"}}\n```\n'
        '```action\n{"tool": "pyrun", "args": {"code": "print(22)"}}\n```'
    )
    turns = [two, "Saw 11 and 22. Done."]
    client = make_client(turns)
    agent = Agent(client, registry, Config(approval="none"))
    session = Session()

    final = agent.run(session, "run both probes")

    assert final == "Saw 11 and 22. Done."
    observations = _observations(session, "pyrun")
    assert len(observations) == 2
    assert any("11" in obs for obs in observations)
    assert any("22" in obs for obs in observations)
    # Both blocks ran inside one model turn, so only two turns were consumed.
    assert len(client.calls) == 2


def test_actions_beyond_the_cap_are_deferred(make_client, registry):
    blocks = "\n".join(
        f'```action\n{{"tool": "pyrun", "args": {{"code": "print({i})"}}}}\n```'
        for i in range(MAX_ACTIONS_PER_TURN + 2)
    )
    turns = [blocks, "Done."]
    client = make_client(turns)
    agent = Agent(client, registry, Config(approval="none"))
    session = Session()

    agent.run(session, "too many at once")

    executed = _observations(session, "pyrun")
    assert len(executed) == MAX_ACTIONS_PER_TURN
    notes = _observations(session, "kat")
    assert any("were not executed" in note for note in notes)


# --- batched actions inside a single block ---------------------------------
#
# The local model often puts several action objects in ONE fenced block,
# comma-separated or as a JSON array, instead of one block each. A brace-greedy
# parser mis-split those on the braces nested in "args" and parsed nothing, so
# the loop spun on "malformed action" forever (seen live on the Cap box). The
# value-by-value decoder reads every object out of the block.


def test_parse_actions_handles_comma_separated_objects_in_one_block():
    text = (
        "```action\n"
        '{"tool": "curl", "args": {"url": "http://h/data/0", "method": "GET"}, "reason": "a"},\n'
        '{"tool": "curl", "args": {"url": "http://h/data/1", "method": "GET"}, "reason": "b"},\n'
        '{"tool": "curl", "args": {"url": "http://h/data/2", "method": "GET"}, "reason": "c"}\n'
        "```"
    )
    actions = parse_actions(text)
    assert [a.tool for a in actions] == ["curl", "curl", "curl"]
    assert [a.args["url"] for a in actions] == [
        "http://h/data/0",
        "http://h/data/1",
        "http://h/data/2",
    ]
    # The nested "args" braces must not have truncated the first object.
    assert actions[0].args == {"url": "http://h/data/0", "method": "GET"}


def test_parse_actions_handles_a_json_array_of_actions():
    text = (
        "```action\n"
        '[{"tool": "a", "args": {"x": 1}}, {"tool": "b", "args": {"y": 2}}]\n'
        "```"
    )
    actions = parse_actions(text)
    assert [a.tool for a in actions] == ["a", "b"]
    assert actions[1].args == {"y": 2}


def test_parse_actions_handles_an_unclosed_fence():
    # The model opened a fence and emitted valid JSON but never closed it.
    text = '```action\n{"tool": "nmap", "args": {"target": "10.0.0.1"}, "reason": "scan"}'
    actions = parse_actions(text)
    assert [a.tool for a in actions] == ["nmap"]


def test_parse_actions_stops_at_trailing_non_json():
    # A stray think-close tag after a valid object must not break parsing.
    text = '```action\n{"tool": "a", "args": {}}\n</think>\n```'
    actions = parse_actions(text)
    assert [a.tool for a in actions] == ["a"]


def test_comma_separated_objects_in_one_block_all_run(make_client, registry):
    batched = (
        "```action\n"
        '{"tool": "pyrun", "args": {"code": "print(11)"}, "reason": "a"},\n'
        '{"tool": "pyrun", "args": {"code": "print(22)"}, "reason": "b"}\n'
        "```"
    )
    turns = [batched, "Saw 11 and 22. Done."]
    client = make_client(turns)
    agent = Agent(client, registry, Config(approval="none"))
    session = Session()

    final = agent.run(session, "run both probes in one block")

    assert final == "Saw 11 and 22. Done."
    observations = _observations(session, "pyrun")
    assert len(observations) == 2
    assert any("11" in obs for obs in observations)
    assert any("22" in obs for obs in observations)
    # Both ran inside one model turn - no malformed-JSON spin.
    assert len(client.calls) == 2
    assert not _observations(session, "kat")


# --- graceful recovery from a context overflow -----------------------------
#
# The local server has a fixed, small token context. If a prompt is rejected
# for exceeding it, the loop trims the window and retries once rather than
# aborting the whole mission on a recoverable sizing error.


class _RaiseThenAnswer:
    """A client that raises a given error on its first call, then answers."""

    def __init__(self, error: Exception, text: str) -> None:
        self._error: Exception | None = error
        self._text = text
        self.calls: list[list[dict]] = []

    def chat(self, messages, *, stream=None, on_token=None, **_kwargs):
        self.calls.append(list(messages))
        if self._error is not None:
            error, self._error = self._error, None
            raise error
        return SimpleNamespace(text=self._text)


class _CapturingReporter:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def event(self, kind: str, message: str) -> None:
        self.events.append((kind, message))

    def token(self, text: str) -> None:
        pass

    def thinking(self) -> None:
        pass


def test_is_context_overflow_recognises_the_endpoint_message():
    assert _is_context_overflow(LLMError("HTTP 400: exceed_context_size_error ..."))
    assert _is_context_overflow(LLMError("the context size was exceeded, try increasing it"))
    assert not _is_context_overflow(LLMError("cannot reach model: connection refused"))


def test_think_retries_once_on_context_overflow(registry):
    overflow = LLMError(
        'model endpoint returned HTTP 400: {"error":{"message":"request (8498 tokens) '
        'exceeds the available context size (8192 tokens)","type":"exceed_context_size_error"}}'
    )
    client = _RaiseThenAnswer(overflow, "Recovered after trimming the window.")
    reporter = _CapturingReporter()
    agent = Agent(client, registry, Config(approval="none"), reporter=reporter)
    session = Session(system="profile")

    final = agent.run(session, "inspect the target")

    assert final == "Recovered after trimming the window."
    # One rejected attempt, then one that succeeded on a smaller window.
    assert len(client.calls) == 2
    assert any(kind == "warn" and "context full" in message for kind, message in reporter.events)


def test_think_does_not_retry_other_errors(registry):
    client = _RaiseThenAnswer(LLMError("cannot reach model: connection refused"), "unused")
    agent = Agent(client, registry, Config(approval="none"))
    session = Session()

    try:
        agent.run(session, "inspect the target")
    except LLMError:
        pass
    else:  # pragma: no cover - the call must propagate the error
        raise AssertionError("a non-overflow LLMError must not be swallowed")

    # It failed on the first attempt without a retry.
    assert len(client.calls) == 1
