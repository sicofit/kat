"""Tests for action parsing, result rendering, and the agent loop.

The loop tests use a scripted client and the interpreter-backed registry, so a
tool genuinely runs through the real executor. The emphasis is on the property
that matters most for this agent: an observation reflects the actual command
output, never text the model produced.
"""

from __future__ import annotations

from kat.agent import Action, Agent, parse_action, render_result
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
